# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import repeat
from numbers import Number
import os
from threading import Thread
import threading
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import zmq

from pymor.core.interfaces import BasicInterface
from pymor.core.pickle import dumps, loads
from pymor.parallel.basic import (WorkerPoolBase, _setup_worker, _store, _remove_object, _get_object, _worker_call_function)
from pymor.parallel.interfaces import RemoteObjectBase


def split_message(message):
    try:
        i_empty = message.index(b'')
    except ValueError:
        raise ValueError('Malformed message: {}'.format(message))
    destination, message = message[:i_empty], message[i_empty+1:]
    try:
        i_empty = message.index(b'')
    except ValueError:
        raise ValueError('Malformed message: {}'.format(message))
    return_path, message = message[:i_empty], message[i_empty+1:]
    return destination, return_path, message


def format_message(destination, return_path, message):
    destination = destination or []
    return_path = return_path or []
    return destination + [b''] + return_path + [b''] + message


class RoutedSocket:

    def send_multipart(self, message):
        if self.recv_state:
            raise ValueError('Not in send state')
        message = format_message(self.path, None, message)
        self.socket.send_multipart(message)
        self.recv_state = True

    def recv_multipart(self):
        if not self.recv_state:
            raise ValueError('Not in receive state')
        message = self.socket.recv_multipart()
        destination, return_path, message = split_message(message)
        assert destination == []
        self.return_path = return_path
        self.recv_state = False
        return message

    def close(self):
        self.socket.close()


class RoutedRequest(RoutedSocket):

    def __init__(self, ctx, address, path):
        self.address, self.path = address, path
        self.socket = ctx.socket(zmq.DEALER)
        self.socket.connect(address)
        self.recv_state = False


class RoutedService(RoutedSocket):

    def __init__(self, ctx, identity, initial_path=None):
        self.socket = ctx.socket(zmq.DEALER)
        self.socket.identity = self.identity = identity
        self.socket.connect('inproc://backend')
        self.path = initial_path
        self.recv_state = initial_path is None

    def recv_multipart(self):
        result = super().recv_multipart()
        self.path = self.return_path
        return result


class ZMQNode(BasicInterface):

    def __init__(self, routing_address, routing_is_child):
        self.routing_address, self.routing_is_child = routing_address, routing_is_child
        self.ctx = zmq.Context()
        self.route_thread = Thread(target=self.route_loop)
        self.route_thread.start()

    def route_loop(self):
        if self.routing_is_child:
            frontend = self.ctx.socket(zmq.DEALER)
            frontend.connect(self.routing_address)
        else:
            frontend = self.ctx.socket(zmq.ROUTER)
            frontend.router_mandatory = 1
            frontend.bind(self.routing_address)

        backend = self.ctx.socket(zmq.ROUTER)
        backend.router_mandatory = 1
        backend.bind('inproc://backend')

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)

        quit = False
        while not quit:
            for socket in dict(poller.poll()):
                message = socket.recv_multipart()
                destination, return_path, message = split_message(message)

                if socket.type == zmq.ROUTER:
                    return_path.insert(0, destination.pop(0))
                if socket == frontend:
                    return_path.insert(0, b'OUT')

                if not destination:
                    assert message == [b'QIT']
                    quit = True
                    continue

                self.logger.debug('Route from={}, to={}, packets={}'.format(return_path, destination, len(message)))

                if destination[0] == b'OUT':
                    destination.pop(0)
                    destination_socket = frontend
                else:
                    destination_socket = backend

                destination_socket.send_multipart(format_message(destination, return_path, message))

        frontend.close()
        backend.close()


class ZMQWorker(ZMQNode):

    def __init__(self, controller_address):
        super().__init__(controller_address, True)
        self.main_thread_id = threading.get_ident()
        ctl_thread = Thread(target=self.ctl_loop, daemon=True)
        ctl_thread.start()
        self.evaluating = False
        self.eval_loop()  # must be main thread in order to abort computations
        self.route_thread.join()
        ctl_thread.join()
        self.ctx.term()

    def ctl_loop(self):
        socket = RoutedService(self.ctx, b'CTL')

        while True:
            cmd, *message = socket.recv_multipart()
            sys.stdout.flush()
            if cmd == b'ABT':
                print('Aborting computation ..')
                sys.stdout.flush()
                if self.evaluating:
                    signal.pthread_kill(self.main_thread_id, signal.SIGINT)
                socket.send_multipart([b'OK'])
            elif cmd == b'QIT':
                print('Shutting down Worker ..')
                sys.stdout.flush()
                socket.send_multipart([b'OK'])
                if self.evaluating:
                    signal.pthread_kill(self.main_thread_id, signal.SIGINT)
                socket.socket.send_multipart(format_message([b'EVL'], None, [b'QIT']))
                socket.socket.send_multipart(format_message([], None, [b'QIT']))
                socket.close()
                break
            else:
                raise NotImplementedError

    def eval_loop(self):
        socket = RoutedService(self.ctx, b'EVL', [b'OUT', b'CTL'])

        socket.send_multipart([b'RWK'])
        message = socket.recv_multipart()
        socket.recv_state = True
        assert message == [b'OK']
        self.logger.info('Registration with controller successful!')

        while True:
            message = socket.recv_multipart()

            if message == [b'QIT']:
                break

            try:
                self.evaluating = True
                if message[0] == b'CMM':
                    src, dst = message[1:]
                    src, dst = loads(src), loads(dst)
                    self.communicate(src, dst)
                    result = b'OK'
                else:
                    assert len(message) == 1, message
                    payload = loads(message[0])
                    f, args, kwargs = payload
                    self.logger.debug('Calling {}'.format(f.__name__))
                    result = f(*args, **kwargs)
                self.evaluating = False
            except KeyboardInterrupt:
                print('interrupted')
                sys.stdout.flush()
                self.evaluating = False
                result = None
            except Exception as e:
                print('Exception raised during function evaluation:')
                traceback.print_exc()
                sys.stdout.flush()
                result = traceback.TracebackException.from_exception(e)
                result.exc_traceback = ''

            socket.send_multipart([dumps(result)])

        socket.close()

    def communicate(self, src, dst):
        socket = self.ctx.socket(zmq.DEALER)
        socket.identity = b'CMM'
        socket.connect('inproc://backend')

        try:
            try:
                src = _get_object(src)
                dst = _get_object(dst)
                if not isinstance(src, dict):
                    raise ValueError('Source not a dictionary.')
                if not isinstance(dst, dict):
                    raise ValueError('Destination not a dictionary.')
            except Exception as e:
                socket.send_multipart(format_message([b'OUT', b'CMM'], None, [b'ERR']))
                raise e

            socket.send_multipart(format_message([b'OUT', b'CMM'], None, [b'RDY']))

            to_send = src.copy()
            self.logger.debug('Communicating {} items'.format(len(to_send)))

            sending = False
            while True:
                message = None
                try:
                    message = socket.recv_multipart(0 if not sending else zmq.NOBLOCK)
                except zmq.ZMQError:
                    pass

                if message:
                    destination, return_path, message = split_message(message)
                    assert destination == []
                    if message == [b'SND']:
                        self.logger.debug('Starting to send.')
                        sending = True
                    elif message == [b'EOC']:
                        break
                    else:
                        worker, value = message
                        worker, value = int(worker), loads(value)
                        dst[worker] = value
                elif sending:
                    if to_send:
                        worker, value = to_send.popitem()
                        assert isinstance(worker, int)
                        message = format_message([b'OUT', b'CMM'], None, [str(worker).encode(), dumps(value)])
                    else:
                        sending = False
                        message = format_message([b'OUT', b'CMM'], None, [b'EOC'])
                    socket.send_multipart(message)

        finally:
            socket.close()


class ZMQController(ZMQNode):

    def __init__(self, address):
        super().__init__(address, False)
        self.connected = False
        cmd_thread = Thread(target=self.cmd_loop)
        cmd_thread.start()
        self.ctl_loop()
        self.route_thread.join()
        cmd_thread.join()
        self.ctx.term()

    def ctl_loop(self):
        self.worker_paths = []
        socket = RoutedService(self.ctx, b'CTL')

        try:
            while True:
                cmd, *args = socket.recv_multipart()
                if cmd == b'RWK':
                    assert not args
                    assert not self.connected
                    self.logger.info('Registered worker: {}'.format(socket.return_path[:-1]))
                    self.worker_paths.append(socket.return_path[:-1])
                    socket.send_multipart([b'OK'])
                elif cmd == b'CON':
                    assert not self.connected
                    self.connected = True
                    self.logger.info('Pool frontend connected')
                    socket.send_multipart([dumps(len(self.worker_paths))])
                elif cmd == b'DSC':
                    assert self.connected
                    self.connected = False
                    self.logger.info('Pool frontend disconnected')
                    socket.send_multipart([b'OK'])
                elif cmd == b'ABT':
                    assert self.connected
                    self.logger.info('Aborting computation')
                    self.call_workers(b'CTL', None, [b'ABT'])
                    socket.send_multipart([b'OK'])
                elif cmd == b'QIT':
                    socket.send_multipart([b'OK'])
                    break
                else:
                    raise NotImplementedError

        except KeyboardInterrupt:
            pass

        self.logger.info('Shutting down workers ...')
        self.call_workers(b'CTL', None, [b'QIT'])
        socket.socket.send_multipart(format_message([b'CMD'], None, [b'QIT']))
        socket.socket.send_multipart(format_message([], None, [b'QIT']))
        socket.close()

    def cmd_loop(self):
        socket = RoutedService(self.ctx, b'CMD')

        while True:
            cmd, *message = socket.recv_multipart()
            if cmd == b'APL':
                assert self.connected
                payload, worker = message
                worker = loads(worker)
                self.logger.debug('Applying function on workers: {}'.format('ALL' if worker is None else worker))
                replies = self.call_workers(b'EVL', worker, [payload])
                socket.send_multipart(replies)
            elif cmd == b'SCT':
                assert self.connected
                self.logger.debug('Scattering data.')
                payload = message
                replies = self.call_workers(b'EVL', None, [[dumps((_store, (loads(p),), {}))] for p in payload])
                assert len(set(replies)) == 1
                socket.send_multipart([replies[0]])
            elif cmd == b'CMM':
                assert self.connected
                self.logger.debug('Worker to worker communication.')
                src, dst = message
                replies = self.communicate(src, dst)
                socket.send_multipart(replies)
            elif cmd == b'QIT':
                break
            else:
                raise NotImplementedError

        socket.close()

    def call_workers(self, service, worker, message):
        if worker is None:
            worker = range(len(self.worker_paths))
        elif isinstance(worker, Number):
            worker = [worker]
        if isinstance(message[0], bytes):
            message = repeat(message)

        socket = self.ctx.socket(zmq.DEALER)
        socket.identity = b'WRK' + service
        socket.connect('inproc://backend')

        for w, msg in zip(worker, message):
            socket.send_multipart(format_message(self.worker_paths[w] + [service], None, msg))

        replies = [None] * len(worker)
        missing = set(worker)

        while missing:
            self.logger.debug('Waiting for replies from workers: {}'.format(list(sorted(missing))))
            destination, return_path, message = split_message(socket.recv_multipart())
            assert destination == []
            assert len(message) == 1
            i = self.worker_paths.index(return_path[:-1])
            replies[worker.index(i)] = message[0]
            missing.remove(i)

        self.logger.debug('Received replies from all workers.')
        socket.close()

        return replies

    def communicate(self, src, dst):
        socket = self.ctx.socket(zmq.DEALER)
        socket.identity = b'CMM'
        socket.connect('inproc://backend')

        self.logger.debug('Entering communcation mode.')
        for w in range(len(self.worker_paths)):
            socket.send_multipart(format_message(self.worker_paths[w] + [b'EVL'], None, [b'CMM', src, dst]))

        self.logger.debug('Waiting for workers to settle.')
        missing = set(range(len(self.worker_paths)))
        failed = set()
        while missing:
            destination, return_path, message = split_message(socket.recv_multipart())
            assert destination == []
            assert message in [[b'RDY'], [b'ERR']], message
            worker = self.worker_paths.index(return_path[:-1])
            missing.remove(worker)
            if message == [b'ERR']:
                failed.add(worker)

        if failed:
            self.logger.critical('Communication error. Aborting.')
            for w in range(len(self.worker_paths)):
                if w not in failed:
                    socket.send_multipart(format_message(self.worker_paths[w] + [b'EVL'], None, [b'CMM', src, dst]))

            replies = [None] * len(self.worker_paths)
            missing = set(range(len(self.worker_paths)))
            while missing:
                destination, return_path, message = split_message(socket.recv_multipart())
                assert destination == []
                assert len(message) == 1
                worker = self.worker_paths.index(return_path[:-1])
                missing.remove(worker)
                replies[worker] = message[0]
            return replies

        self.logger.debug('Starting communcation.')
        for w in range(len(self.worker_paths)):
            socket.send_multipart(format_message(self.worker_paths[w] + [b'CMM'], None, [b'SND']))

        sending = set(range(len(self.worker_paths)))
        while sending:
            self.logger.debug('Waiting for workers: {}'.format(list(sorted(sending))))
            destination, return_path, message = split_message(socket.recv_multipart())
            assert destination == []
            if len(message) == 1:
                assert message == [b'EOC'], (return_path, message)
                sending.remove(self.worker_paths.index(return_path[:-1]))
            else:
                assert len(message) == 2
                dst_worker, value = message
                dst_worker = int(dst_worker)
                assert 0 <= dst_worker < len(self.worker_paths), dst_worker
                src_worker = self.worker_paths.index(return_path[:-1])
                message = format_message(self.worker_paths[dst_worker] + [b'CMM'], None,
                                         [str(src_worker).encode(), value])
                socket.send_multipart(message)

        self.logger.debug('Received replies from all workers.')

        self.logger.debug('Leaving communication mode.')
        for w in range(len(self.worker_paths)):
            socket.send_multipart(format_message(self.worker_paths[w] + [b'CMM'], None, [b'EOC']))

        missing = set(range(len(self.worker_paths)))
        while missing:
            destination, return_path, message = split_message(socket.recv_multipart())
            assert destination == []
            assert len(message) == 1 and loads(message[0]) == b'OK', message
            worker = self.worker_paths.index(return_path[:-1])
            missing.remove(worker)

        self.logger.debug('Done.')
        socket.close()
        return [b'OK']


class RemoteException(Exception):

    def __init__(self, result):
        self.result = result

    def __str__(self):
        msg = ''
        for i_r, r in enumerate(self.result):
            if isinstance(r, traceback.TracebackException):
                msg += '\n\nException on worker {}:\n'.format(i_r)
                msg += '----------------------\n'
                for line in r.format():
                    msg += line
        return msg


class new_zmq_pool(BasicInterface):

    def __init__(self, num_workers=None, wait=3):
        self.num_workers = num_workers
        self.wait = wait

    def __enter__(self):
        self.logger.info('Spawning controller and {} workers'.format(self.num_workers))

        self.tempdir = tempfile.mkdtemp()
        path = 'ipc://' + os.path.join(self.tempdir, 'zmqpool')

        for _ in range(self.num_workers):
            subprocess.Popen('python -m pymor.parallel.zmq worker ' + path,
                             start_new_session=True, shell=True)
        subprocess.Popen('python -m pymor.parallel.zmq controller ' + path,
                         start_new_session=True, shell=True)

        self.logger.info('Waiting for workers to connect ...')
        time.sleep(self.wait)

        self.pool = ZMQPool(path)
        return self.pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.shutdown()
        shutil.rmtree(self.tempdir)


class ZMQPool(WorkerPoolBase):

    def __init__(self, controller_address, ctx=None):
        super().__init__()
        self.controller_address = controller_address
        self.ctx = ctx or zmq.Context()
        self.connected = False
        self.connect()

    def connect(self):
        assert not self.connected
        self.logger.info('Connecting to controller at {}'.format(self.controller_address))
        self.control_socket = RoutedRequest(self.ctx, self.controller_address, [b'CTL'])
        self.command_socket = RoutedRequest(self.ctx, self.controller_address, [b'CMD'])
        self.control_socket.send_multipart([b'CON'])
        self.size = loads(self.control_socket.recv_multipart()[0])
        self.command_socket.send_multipart([b'APL', dumps((_setup_worker, (), {})), dumps(None)])
        self.command_socket.recv_multipart()
        self.logger.info('Connected to {} workers'.format(self.size))
        self.connected = True

    def disconnect(self):
        assert self.connected
        self.logger.info('Disconnecting from controller ...')
        self.command_socket.send_multipart([b'APL', dumps((_setup_worker, (), {})), dumps(None)])  # clear state
        self.command_socket.recv_multipart()
        self.control_socket.send_multipart([b'DSC'])
        reply = self.control_socket.recv_multipart()
        assert reply == [b'OK']
        self.command_socket.close()
        self.control_socket.close()
        self.logger.info('Disconnected')
        self.connected = False

    def shutdown(self):
        self.disconnect()
        control_socket = RoutedRequest(self.ctx, self.controller_address, [b'CTL'])
        control_socket.send_multipart([b'QIT'])
        control_socket.recv_multipart()
        control_socket.close()
        self.ctx.term()

    def __len__(self):
        assert self.connected
        return self.size

    def _scatter(self, l):
        assert self.connected
        self.command_socket.send_multipart([b'SCT'] + [dumps(x) for x in l])
        result = self.command_socket.recv_multipart()
        assert len(set(result)) == 1
        return loads(result[0])

    def _remove(self, remote_resource):
        assert self.connected
        self.command_socket.send_multipart([b'APL', dumps((_remove_object, (remote_resource,), {})), dumps(None)])
        self.command_socket.recv_multipart()

    def _apply(self, function, *args, store=False, worker=None, **kwargs):
        assert self.connected
        assert worker is None or not store

        self.command_socket.send_multipart([b'APL',
                                            dumps((_worker_call_function, (function, args, kwargs, store), {})),
                                            dumps(worker)])
        try:
            result = self.command_socket.recv_multipart()
            result = [loads(r) for r in result]
            if any(isinstance(r, traceback.TracebackException) for r in result):
                raise RemoteException(result)
        except KeyboardInterrupt as e:
            self.control_socket.send_multipart([b'ABT'])
            self.control_socket.recv_multipart()
            result = self.command_socket.recv_multipart()
            raise e

        if store:
            assert len(set(result)) == 1
            return result[0]
        elif isinstance(worker, Number):
            return result[0]
        else:
            return result

    def communicate(self, source, destination):
        assert self.connected
        assert isinstance(source, RemoteObjectBase)
        assert isinstance(destination, RemoteObjectBase)

        source = self._map_obj(source)
        destination = self._map_obj(destination)

        self.command_socket.send_multipart([b'CMM', dumps(source), dumps(destination)])

        try:
            result = self.command_socket.recv_multipart()
            if not result == [b'OK']:
                result = [loads(r) for r in result]
                raise RemoteException(result)
        except KeyboardInterrupt as e:
            self.control_socket.send_multipart([b'ABT'])
            self.control_socket.recv_multipart()
            result = self.command_socket.recv_multipart()
            raise e

    def __del__(self):
        if self.connected:
            self.disconnect()


if __name__ == '__main__':
    import docopt
    from pymor.parallel.zmq import ZMQController, ZMQWorker

    args = docopt.docopt("""
Usage:
    zmq.py controller [CONTROLLER_ADDRESS]
    zmq.py worker     CONTROLLER_ADDRESS
""")

    if args['controller']:
        ZMQController(args['CONTROLLER_ADDRESS'] or 'tcp://*:5555')
    elif args['worker']:
        ZMQWorker(args['CONTROLLER_ADDRESS'])
