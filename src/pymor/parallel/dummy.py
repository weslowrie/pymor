# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from copy import deepcopy
from numbers import Number

from pymor.core.interfaces import ImmutableInterface
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

    def _apply(self, function, *args, store=False, worker=None, **kwargs):
        assert worker is None or not store

        if worker is None:
            worker, single_worker = range(self.size), False
        elif isinstance(worker, Number):
            worker, single_worker = [worker], True
        else:
            worker, single_worker = worker, False

        result = []

        for worker_id in worker:
            if isinstance(function, RemoteResourceWithPath):
                f = function.resolve_path(function.remote_resource[worker_id])
            else:
                f = function

            worker_args = [(v.resolve_path(v.remote_resource[worker_id]) if isinstance(v, RemoteResourceWithPath) else
                            self._copy(v)) for v in args]
            worker_kwargs = {k: (v.resolve_path(v.remote_resource[worker_id]) if isinstance(v, RemoteResourceWithPath) else
                                 self._copy(v)) for k, v in kwargs.items()}

            result.append(f(*worker_args, **worker_kwargs))

        if single_worker:
            result = result[0]

        return result

    def _scatter(self, l):
        return [self._copy(x) for x in l]

    def _copy(self, obj):
        return obj if isinstance(obj, ImmutableInterface) else deepcopy(obj)


dummy_pool = DummyPool()
