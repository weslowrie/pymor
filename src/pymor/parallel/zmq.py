# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import repeat
from numbers import Number
import signal
from threading import Thread
import threading
import sys
import time
import traceback

import zmq

from pymor.core.pickle import dumps, loads
from pymor.parallel.basic import (WorkerPoolBase, _setup_worker, _store, _remove_object, _worker_call_function)


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


def route(frontend, backend):
    assert frontend.type in [zmq.ROUTER, zmq.DEALER]
    assert backend.type == zmq.ROUTER
    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)
    poller.register(backend, zmq.POLLIN)

    while True:
        for socket in dict(poller.poll()):
            message = socket.recv_multipart()
            destination, return_path, message = split_message(message)

            if socket.type == zmq.ROUTER:
                return_path.insert(0, destination.pop(0))
            if socket == frontend:
                return_path.insert(0, b'OUT')

            print('ROUTE:', destination, return_path, str(message)[:32])

            if destination[0] == b'OUT':
                destination.pop(0)
                destination_socket = frontend
            else:
                destination_socket = backend

            destination_socket.send_multipart(format_message(destination, return_path, message))


class RoutedRequest:

    def __init__(self, ctx, address, path):
        self.address, self.path = address, path
        self.socket = ctx.socket(zmq.DEALER)
        self.socket.connect(address)
        self.recv_state = False

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
        assert return_path == self.path
        self.recv_state = False
        return message


class RoutedService:

    def __init__(self, ctx, identity, initial_return_path=None):
        self.socket = ctx.socket(zmq.DEALER)
        self.socket.identity = self.identity = identity
        self.socket.connect('inproc://backend')
        self.return_path = initial_return_path
        self.recv_state = initial_return_path is None

    def recv_multipart(self):
        if not self.recv_state:
            raise ValueError('Not in receive state')
        message = self.socket.recv_multipart()
        destination, return_path, message = split_message(message)
        assert destination == []
        self.return_path = return_path
        self.recv_state = False
        return message

    def send_multipart(self, message):
        if self.recv_state:
            raise ValueError('Not in send state')
        message = format_message(self.return_path, None, message)
        self.socket.send_multipart(message)
        self.recv_state = True


class ZMQWorker:

    def __init__(self, controller_address):
        self.controller_address = controller_address
        self.ctx = zmq.Context()
        self.main_thread_id = threading.get_ident()
        Thread(target=self.main_loop).start()
        Thread(target=self.ctl_loop).start()
        self.eval_loop()

    def main_loop(self):
        frontend = self.ctx.socket(zmq.DEALER)
        frontend.connect(self.controller_address)
        backend = self.ctx.socket(zmq.ROUTER)
        backend.bind('inproc://backend')
        route(frontend, backend)

    def ctl_loop(self):
        socket = RoutedService(self.ctx, b'CTL')

        while True:
            cmd, *message = socket.recv_multipart()
            sys.stdout.flush()
            if cmd == b'ABT':
                print('Aborting computation ..')
                sys.stdout.flush()
                signal.pthread_kill(self.main_thread_id, signal.SIGINT)
                socket.recv_state = True  # no reply
            else:
                raise NotImplementedError

    def eval_loop(self):
        socket = RoutedService(self.ctx, b'EVL', [b'OUT', b'CTL'])

        socket.send_multipart([b'RWK'])
        message = socket.recv_multipart()
        socket.recv_state = True
        assert message == [b'OK']
        print('Registration with controller successful!')

        while True:
            message = socket.recv_multipart()
            assert len(message) == 1
            payload = loads(message[0])

            f, args, kwargs = payload
            print(f)
            try:
                result = f(*args, **kwargs)
            except KeyboardInterrupt:
                print('interrupted')
                sys.stdout.flush()
                result = None
            except Exception as e:
                print('Exception raised during function evaluation:')
                traceback.print_exc()
                sys.stdout.flush()
                result = traceback.TracebackException.from_exception(e)
                result.exc_traceback = ''

            socket.send_multipart([dumps(result)])


class ZMQController:

    def __init__(self, address):
        self.address = address
        self.ctx = zmq.Context()
        self.connected = False
        Thread(target=self.ctl_loop).start()
        Thread(target=self.main_loop).start()
        self.cmd_loop()

    def main_loop(self):
        backend = self.ctx.socket(zmq.ROUTER)
        backend.bind('inproc://backend')
        time.sleep(1)
        frontend = self.ctx.socket(zmq.ROUTER)
        frontend.bind(self.address)
        route(frontend, backend)

    def ctl_loop(self):
        self.worker_paths = []
        socket = RoutedService(self.ctx, b'CTL')
        worker_socket = self.ctx.socket(zmq.DEALER)
        worker_socket.identity = b'WRKCTL'
        worker_socket.connect('inproc://backend')

        while True:
            cmd, *args = socket.recv_multipart()
            print(cmd)
            if cmd == b'RWK':
                assert not args
                assert not self.connected
                self.worker_paths.append(socket.return_path[:-1])
                socket.send_multipart([b'OK'])
            elif cmd == b'CON':
                assert not self.connected
                self.connected = True
                socket.send_multipart([dumps(len(self.worker_paths))])
            elif cmd == b'DSC':
                assert self.connected
                self.connected = False
                socket.send_multipart([b'OK'])
            elif cmd == b'ABT':
                assert self.connected
                for w in self.worker_paths:
                    worker_socket.send_multipart(format_message(w + [b'CTL'], None, [b'ABT']))
                socket.send_multipart([b'OK'])
            else:
                raise NotImplementedError

    def cmd_loop(self):
        socket = RoutedService(self.ctx, b'CMD')

        while True:
            cmd, *message = socket.recv_multipart()
            print(cmd)
            if cmd == b'APL':
                assert self.connected
                payload, worker = message
                worker = loads(worker)
                replies = self.call_workers(worker, payload)
                socket.send_multipart(replies)
            elif cmd == b'SCT':
                assert self.connected
                payload = message
                replies = self.call_workers(None, [dumps((_store, (loads(p),), {})) for p in payload])
                assert len(set(replies)) == 1
                socket.send_multipart([replies[0]])
            else:
                raise NotImplementedError

    def call_workers(self, worker, message):
        socket = self.ctx.socket(zmq.DEALER)
        socket.identity = b'WRK'
        socket.connect('inproc://backend')

        if worker is None:
            worker = range(len(self.worker_paths))
        elif isinstance(worker, Number):
            worker = [worker]
        if isinstance(message, bytes):
            message = repeat(message)

        print('Sending messages to workers ...')
        for w, msg in zip(worker, message):
            socket.send_multipart(format_message(self.worker_paths[w] + [b'EVL'], None, [msg]))

        print('Waiting for replies ', end='')
        replies = [None] * len(worker)
        missing = set(worker)

        while missing:
            destination, return_path, message = split_message(socket.recv_multipart())
            assert destination == []
            assert len(message) == 1
            i = self.worker_paths.index(return_path[:-1])
            replies[worker.index(i)] = message[0]
            missing.remove(i)
            print(i, end='')
            sys.stdout.flush()

        return replies


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
        self.logger.info('Disconnected')
        self.connected = False

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
        except KeyboardInterrupt:
            self.control_socket.send_multipart([b'ABT'])
            self.control_socket.recv_multipart()
            result = self.command_socket.recv_multipart()
            raise KeyboardInterrupt

        if store:
            assert len(set(result)) == 1
            return loads(result[0])
        elif isinstance(worker, Number):
            return loads(result[0])
        else:
            return [loads(r) for r in result]

    def __del__(self):
        if self.connected:
            self.disconnect()


if __name__ == '__main__':
    import docopt
    args = docopt.docopt("""
Usage:
    zmq.py controller [CONTROLLER_ADDRESS]
    zmq.py worker     CONTROLLER_ADDRESS
""")

    if args['controller']:
        ZMQController(args['CONTROLLER_ADDRESS'] or 'tcp://*:5555')
    elif args['worker']:
        ZMQWorker(args['CONTROLLER_ADDRESS'])
