# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

from pymor.core.interfaces import ImmutableInterface
from pymor.core.pickle import dumps, loads
from pymor.parallel.basic import WorkerPoolBase, RemoteResourceWithPath


class DummyPool(WorkerPoolBase):

    def __init__(self, size=1):
        self.size = size
        super().__init__()

    def __len__(self):
        return self.size

    def __bool__(self):
        return self.size > 1

    def _remove(self, remote_resource):
        pass

    def _apply(self, function, *args, store=False, scatter=False, worker=None, **kwargs):
        assert worker is None or (not store and not scatter)

        if worker is None:
            worker, single_worker = range(self.size), False
        elif isinstance(worker, Number):
            worker, single_worker = [worker], True
        else:
            worker, single_worker = worker, False

        result = []

        if scatter:
            args = zip(*args) if args else [()] * len(worker)
        else:
            args = [args] * len(worker)

        for worker_id, worker_args in zip(worker, args):
            if isinstance(function, RemoteResourceWithPath):
                f = function.resolve_path(function.remote_resource[worker_id])
            else:
                f = function

            worker_args = [(v if isinstance(v, ImmutableInterface) else
                            v.resolve_path(v.remote_resource[worker_id]) if isinstance(v, RemoteResourceWithPath) else
                            self._copy(v)) for v in worker_args]
            worker_kargs = {k: (v if isinstance(v, ImmutableInterface) else
                                v.resolve_path(v.remote_resource[worker_id]) if isinstance(v, RemoteResourceWithPath) else
                                self._copy(v)) for k, v in kwargs.items()}

            result.append(f(*worker_args, **worker_kargs))

        if single_worker:
            result = result[0]

        return result

    def _copy(self, obj):
        return loads(dumps(obj))


dummy_pool = DummyPool()
