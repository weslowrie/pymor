# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number
import os
import time

from pymor.core.config import config
from pymor.core.interfaces import BasicInterface
from pymor.parallel.basic import WorkerPoolBase, RemoteResourceWithPath
from pymor.tools.counter import Counter


if config.HAVE_IPYTHON:
    try:
        from ipyparallel import Client, TimeoutError
    except ImportError:
        from IPython.parallel import Client, TimeoutError


class new_ipcluster_pool(BasicInterface):
    """Create a new IPython parallel cluster and connect to it.

    This context manager can be used to create an :class:`IPythonPool`
    |WorkerPool|. When entering the context a new IPython cluster is
    created using the `ipcluster` script and an :class:`IPythonPool`
    is instantiated for the newly created cluster. When leaving
    the context the cluster is shut down.

    Parameters
    ----------
    profile
        Passed as `--profile` parameter to the `ipcluster` script.
    cluster_id
        Passed as `--cluster-id` parameter to the `ipcluster` script.
    nun_engines
        Passed as `--n` parameter to the `ipcluster` script.
    ipython_dir
        Passed as `--ipython-dir` parameter to the `ipcluster` script.
    min_wait
        Wait at least this many seconds before trying to connect to the
        new cluster.
    timeout
        Wait at most this many seconds for all Ipython cluster engines to
        become available.
    """

    def __init__(self, profile=None, cluster_id=None, num_engines=None, ipython_dir=None, min_wait=1, timeout=60):
        self.profile = profile
        self.cluster_id = cluster_id
        self.num_engines = num_engines
        self.ipython_dir = ipython_dir
        self.min_wait = min_wait
        self.timeout = timeout

    def __enter__(self):
        args = []
        if self.profile is not None:
            args.append('--profile=' + self.profile)
        if self.cluster_id is not None:
            args.append('--cluster-id=' + self.cluster_id)
        if self.num_engines is not None:
            args.append('--n=' + str(self.num_engines))
        if self.ipython_dir is not None:
            args.append('--ipython-dir=' + self.ipython_dir)
        cmd = ' '.join(['ipcluster start --daemonize'] + args)
        self.logger.info('Staring IPython cluster with "' + cmd + '"')
        os.system(cmd)

        num_engines, timeout = self.num_engines, self.timeout
        time.sleep(self.min_wait)
        waited = self.min_wait
        client = None
        while client is None:
            try:
                client = Client(profile=self.profile, cluster_id=self.cluster_id)
            except (IOError, TimeoutError):
                if waited >= self.timeout:
                    raise IOError('Could not connect to IPython cluster controller')
                if waited % 10 == 0:
                    self.logger.info('Waiting for controller to start ...')
                time.sleep(1)
                waited += 1

        if num_engines is None:
            while len(client) == 0 and waited < timeout:
                if waited % 10 == 0:
                    self.logger.info('Waiting for engines to start ...')
                time.sleep(1)
                waited += 1
            if len(client) == 0:
                raise IOError('IPython cluster engines failed to start')
            wait = min(waited, timeout - waited)
            if wait > 0:
                self.logger.info('Waiting {} more seconds for engines to start ...'.format(wait))
                time.sleep(wait)
        else:
            running = len(client)
            while running < num_engines and waited < timeout:
                if waited % 10 == 0:
                    self.logger.info('Waiting for {} of {} engines to start ...'
                                     .format(num_engines - running, num_engines))
                time.sleep(1)
                waited += 1
                running = len(client)
            running = len(client)
            if running < num_engines:
                raise IOError('{} of {} IPython cluster engines failed to start'
                              .format(num_engines - running, num_engines))
        client.close()

        self.pool = IPythonPool(profile=self.profile, cluster_id=self.cluster_id)
        return self.pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.client.close()
        args = []
        if self.profile is not None:
            args.append('--profile=' + self.profile)
        if self.cluster_id is not None:
            args.append('--cluster-id=' + self.cluster_id)
        if self.ipython_dir is not None:
            args.append('--ipython-dir=' + self.ipython_dir)
        cmd = ' '.join(['ipcluster stop'] + args)
        self.logger.info('Stopping IPython cluster with "' + cmd + '"')
        os.system(cmd)


class IPythonPool(WorkerPoolBase):
    """|WorkerPool| based on the IPython parallel computing features.

    Parameters
    ----------
    num_engines
        Number of IPython engines to use. If `None`, all available
        engines are used.
    kwargs
        Keyword arguments used to instantiate the IPython cluster client.
    """

    def __init__(self, num_engines=None, **kwargs):
        super().__init__()
        self.client = Client(**kwargs)
        if num_engines is not None:
            self.view = self.client[:num_engines]
        else:
            self.view = self.client[:]
        self.logger.info('Connected to {} engines'.format(len(self.view)))
        self.view.apply_sync(_setup_worker)
        self._remote_objects_created = Counter()

    def __len__(self):
        return len(self.view)

    def _scatter(self, l):
        remote_resource = self._remote_objects_created.inc()
        self.view.map_sync(_store, l, [remote_resource] * len(self))
        return remote_resource

    def _remove(self, remote_resource):
        self.view.apply(_remove_object, remote_resource)

    def _apply(self, function, *args, store=False, worker=None, **kwargs):
        assert worker is None or not store

        if worker is None:
            view = self.view
        elif isinstance(worker, Number):
            view = self.client[int(worker)]
        else:
            view = self.client[[int(w) for w in worker]]

        if store:
            remote_resource = self._remote_objects_created.inc()
        else:
            remote_resource = None

        result = view.apply_sync(_worker_call_function, function, args, kwargs, remote_resource)

        if store:
            return remote_resource
        else:
            return result


def _setup_worker():
    global _remote_objects
    _remote_objects = {}


def _remove_object(remote_resource):
    global _remote_objects
    del _remote_objects[remote_resource]


def _store(obj, remote_resource):
    _remote_objects[remote_resource] = obj


def _worker_call_function(function, args, kwargs, remote_resource):
    global _remote_objects

    def get_obj(obj):
        if isinstance(obj, RemoteResourceWithPath):
            return obj.resolve_path(_remote_objects[obj.remote_resource])
        else:
            return obj

    function = get_obj(function)
    args = (get_obj(v) for v in args)
    kwargs = {k: get_obj(v) for k, v in kwargs.items()}

    result = function(*args, **kwargs)
    if remote_resource is not None:
        _remote_objects[remote_resource] = result
        return None
    else:
        return result
