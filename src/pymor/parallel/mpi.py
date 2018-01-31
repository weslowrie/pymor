# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import chain
from numbers import Number


from pymor.parallel.basic import WorkerPoolBase, RemoteResourceWithPath
from pymor.tools import mpi


class MPIPool(WorkerPoolBase):
    """|WorkerPool| based pyMOR's MPI :mod:`event loop <pymor.tools.mpi>`."""

    def __init__(self):
        super().__init__()
        self.logger.info('Connected to {} ranks'.format(mpi.size))
        self._payload = mpi.call(mpi.function_call_manage, _setup_worker)

    def __del__(self):
        mpi.call(mpi.remove_object, self._payload)

    def __len__(self):
        return mpi.size

    def _apply(self, function, *args, store=False, scatter=False, worker=None, **kwargs):
        assert worker is None or (not store and not scatter)

        payload = mpi.get_object(self._payload)

        payload[0] = (function, args, kwargs)
        if worker is None:
            result = mpi.call(_worker_call_function, store, scatter, payload)
        else:
            result = mpi.call(_single_worker_call_function,
                              [worker] if isinstance(worker, Number) else worker,
                              payload)
            if isinstance(worker, Number):
                result = result[0]
        payload[0] = 0

        return result

    def _remove(self, remote_resource):
        mpi.call(mpi.remove_object, remote_resource)


def _setup_worker():
    return [None]


def _worker_call_function(store, scatter, payload):
    if scatter:
        function, kwargs = mpi.comm.bcast((payload[0][0], payload[0][2]) if mpi.rank0 else None, root=0)
        args = mpi.comm.scatter(zip(*payload[0][1]) if mpi.rank0 else None, root=0)
    else:
        function, args, kwargs = mpi.comm.bcast(payload[0] if mpi.rank0 else None, root=0)

    result = _eval_function(function, args, kwargs)

    if store:
        return mpi.manage_object(result)
    else:
        return mpi.comm.gather(result, root=0)


def _single_worker_call_function(worker, payload):
    if mpi.rank0:
        for w in worker:
            if w == 0:
                pass
            else:
                mpi.comm.send(payload[0], dest=w)

        results = []
        for w in worker:
            if w == 0:
                function, args, kwargs = payload[0]
                result = _eval_function(function, args, kwargs)
            else:
                result = mpi.comm.recv(source=w)
            results.append(result)

        return results

    elif mpi.rank in worker:
        function, args, kwargs = mpi.comm.recv(source=0)
        result = _eval_function(function, args, kwargs)
        mpi.comm.send(result, dest=0)


def _eval_function(function, args, kwargs):
    def get_obj(obj):
        if isinstance(obj, RemoteResourceWithPath):
            return obj.resolve_path(mpi.get_object(obj.remote_resource))
        else:
            return obj

    function = get_obj(function)
    args = (get_obj(v) for v in args)
    kwargs = {k: get_obj(v) for k, v in kwargs.items()}
    return function(*args, **kwargs)
