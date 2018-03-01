# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import chain
from copy import deepcopy
from numbers import Number

from pymor.core.interfaces import ImmutableInterface
from pymor.parallel.basic import WorkerPoolBase, RemoteResourceWithPath
from pymor.parallel.interfaces import RemotePath, RemoteObjectBase


class VirtualPool(WorkerPoolBase):

    def __init__(self, pool, size):
        if len(pool) > size:
            raise NotImplementedError
        super().__init__()
        self.size = size
        self.pool = pool
        self.local_sizes = local_sizes = ([size // len(pool) + 1] * (size % len(pool)) +
                                          [size // len(pool)] * (len(pool) - size % len(pool)))
        self.local_sizes_r = pool.scatter(local_sizes)
        pos = 0
        self.local_to_global_ids = local_to_global_ids = []
        self.global_to_local_ids = global_to_local_ids = []
        for i, s in enumerate(local_sizes):
            local_to_global_ids.append(list(range(pos, pos+s)))
            for j in range(s):
                global_to_local_ids.append((i, j))
            pos += s
        self.local_to_global_ids_r = pool.push(local_to_global_ids)
        self.global_to_local_ids_r = pool.push(global_to_local_ids)

    def __len__(self):
        return self.size

    def _remove(self, remote_resource):
        pass

    def _scatter(self, l):
        slices = []
        ind = 0
        for s in self.local_sizes:
            slices.append(l[ind:ind+s])
            ind += s

        slices = self.pool.apply(DistributedObjectWithPath, self.pool.scatter(slices), store=True)
        return slices

    def _apply(self, function, *args, store=False, worker=None, **kwargs):
        if isinstance(function, RemoteResourceWithPath):
            function = RemotePath(function.remote_resource, function.path)

        args = [RemotePath(v.remote_resource, v.path)
                if isinstance(v, RemoteResourceWithPath) else v for v in args]
        kwargs = {k: (RemotePath(v.remote_resource, v.path)
                      if isinstance(v, RemoteResourceWithPath) else v)
                  for k, v in kwargs.items()}

        if worker is not None:
            if isinstance(worker, Number):
                worker = [worker]
                single_worker = True
            else:
                single_worker = False

            local_worker = [[] for _ in range(len(self.pool))]
            for w in worker:
                local_id = self.global_to_local_ids[w]
                local_worker[local_id[0]].append(local_id[1])

            worker = [i for i, lw in enumerate(local_worker) if lw]
            local_worker = self.pool.scatter(local_worker)
            assert not single_worker or len(worker) == 1
        else:
            single_worker = False
            local_worker = self.local_sizes_r
        assert worker is None or not store

        result = self.pool.apply(_apply, *args, function_=function, store_=store, worker_=local_worker,
                                 store=store, worker=worker, **kwargs)

        if single_worker:
            return result[0][0]
        elif store:
            return result
        else:
            return list(chain(*result))

    def communicate(self, source, destination):
        assert isinstance(source, RemoteObjectBase)
        assert isinstance(destination, RemoteObjectBase)

        source = self._map_obj(source)
        source = RemotePath(source.remote_resource, source.path)
        destination = self._map_obj(destination)
        destination = RemotePath(destination.remote_resource, destination.path)

        physical_src = self.pool.apply(_prepare_communication, source, self.local_sizes_r, self.global_to_local_ids_r,
                                       store=True)
        physical_dst = self.pool.push({})
        self.pool.communicate(physical_src, physical_dst)
        self.pool.apply(_finish_communication, physical_dst, destination, self.local_sizes_r, self.local_to_global_ids_r)


class DistributedObjectWithPath:

    def __init__(self, l, path=[]):
        self.l, self.path = l, path

    def __getattr__(self, name):
        return DistributedObjectWithPath(self.l, self.path + [name])

    def __getitem__(self, key):
        return DistributedObjectWithPath(self.l, self.path + [[key]])

    def resolve_path(self, i):
        o = self.l[i]
        for p in self.path:
            if isinstance(p, list):
                o = o[p[0]]
            else:
                o = getattr(o, p)
        return o


def _apply(*args, function_=None, store_=False, worker_=None, **kwargs):

    def get_obj(obj, i):
        if isinstance(obj, DistributedObjectWithPath):
            return obj.resolve_path(i)
        elif isinstance(obj, ImmutableInterface):
            return obj
        else:
            return deepcopy(obj)

    result = []
    for w in range(worker_) if isinstance(worker_, Number) else worker_:
        worker_function = get_obj(function_, w)
        worker_args = (get_obj(v, w) for v in args)
        worker_kwargs = {k: get_obj(v, w) for k, v in kwargs.items()}

        result.append(worker_function(*worker_args, **worker_kwargs))

    if store_:
        return DistributedObjectWithPath(result)
    else:
        return result


def _prepare_communication(src, worker_, global_to_local_ids):
    assert isinstance(src, DistributedObjectWithPath)

    physical_src = {}
    for w in range(worker_):
        s = src.resolve_path(w)
        if not isinstance(s, dict):
            raise ValueError('Source not a dictionary.')
        for d, v in s.items():
            physical_id, local_id = global_to_local_ids[d]
            if physical_id not in physical_src:
                physical_src[physical_id] = {}
            if local_id not in physical_src[physical_id]:
                physical_src[physical_id][local_id] = {}
            physical_src[physical_id][local_id][w] = v

    return physical_src


def _finish_communication(physical_dst, dst, worker_, local_to_global_ids):
    assert isinstance(dst, DistributedObjectWithPath)
    dsts = [dst.resolve_path(w) for w in range(worker_)]
    assert all(isinstance(d, dict) for d in dsts)

    for src_physical_id, d in physical_dst.items():
        for dst_local_id, dd in d.items():
            for src_local_id, v in dd.items():
                dsts[dst_local_id][local_to_global_ids[src_physical_id][src_local_id]] = v
