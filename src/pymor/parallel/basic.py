# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains a base class for implementing WorkerPoolInterface."""

from itertools import chain
import weakref

from pymor.core.interfaces import ImmutableInterface, abstractmethod
from pymor.parallel.interfaces import WorkerPoolInterface, RemoteObject, RemotePath


class WorkerPoolBase(WorkerPoolInterface):

    def __init__(self):
        self.remote_objects = weakref.WeakSet()
        self._pushed_immutable_objects = weakref.WeakValueDictionary()

    @abstractmethod
    def _apply(self, function, *args, store=False, worker=None, **kwargs):
        pass

    @abstractmethod
    def _scatter(self, l):
        pass

    @abstractmethod
    def _remove(self, remote_resource):
        pass

    def push(self, obj):
        if isinstance(obj, ImmutableInterface):
            if obj.uid in self._pushed_immutable_objects:
                return self._pushed_immutable_objects[obj.uid]

        remote_object = self.apply(_identity, obj, store=True)

        if isinstance(obj, ImmutableInterface):
            self._pushed_immutable_objects[obj.uid] = remote_object

        return remote_object

    def get(self, obj, worker=None):
        return self.apply(_identity, obj, worker=worker)

    def apply(self, function, *args, store=False, worker=None, **kwargs):
        assert worker is None or not store
        function, args, kwargs = self._map_args(function, args, kwargs, False)
        result = self._apply(function, *args, store=store, worker=worker, **kwargs)
        if store:
            result = RemoteObject(self, result)
            self.remote_objects.add(result)
            weakref.finalize(result, self._remove, result.remote_resource)
        return result

    def map(self, function, *args, **kwargs):
        assert len(set(len(a) for a in args)) == 1
        results = self.apply(_map, self.scatter(list(zip(*args)), slice=True), f=function, **kwargs)
        results = list(chain(*results))
        return results

    def scatter(self, l, slice=False):
        if slice:
            slice_len = len(l) // len(self) + (1 if len(l) % len(self) else 0)
            slices = []
            for i in range(len(self)):
                slices.append(l[i*slice_len:(i+1)*slice_len])
            remote_resource = self._scatter(slices)
        else:
            assert len(l) == len(self)
            remote_resource = self._scatter(l)
        remote_object = RemoteObject(self, remote_resource)
        self.remote_objects.add(remote_object)
        weakref.finalize(remote_object, self._remove, remote_resource)
        return remote_object

    def _map_args(self, function, args, kwargs, scatter=False):

        mapped_function = self._map_obj(function)
        mapped_args = args if scatter else tuple(self._map_obj(o) for o in args)
        mapped_kwargs = {k: self._map_obj(v) for k, v in kwargs.items()}

        return mapped_function, mapped_args, mapped_kwargs

    def _map_obj(self, o):
        if isinstance(o, ImmutableInterface) and o.uid in self._pushed_immutable_objects:
            return RemoteResourceWithPath(self._pushed_immutable_objects[o.uid].remote_resource)
        elif isinstance(o, RemoteObject):
            return RemoteResourceWithPath(o.remote_resource)
        elif isinstance(o, RemotePath):
            return RemoteResourceWithPath(o.remote_object.remote_resource, o.path)
        else:
            return o


class RemoteResourceWithPath:

    def __init__(self, remote_resource, path=None):
        self.remote_resource, self.path = remote_resource, path

    def __repr__(self):
        return 'RemoteResourceWithPath({}, {})'.format(self.remote_resource, self.path)

    def resolve_path(self, obj):
        if self.path is None:
            return obj
        o = obj
        for p in self.path:
            if isinstance(p, list):
                o = o[p[0]]
            else:
                o = getattr(o, p)
        return o


def _identity(obj):
    return obj


def _map(chunks, f=None, **kwargs):
    result = [f(*args, **kwargs) for args in chunks]
    return result


######################################################


class RemoteResource(int):
    pass


def _setup_worker():
    global _remote_objects_created, _remote_objects
    _remote_objects_created = 0
    _remote_objects = {}


def _remove_object(remote_resource):
    global _remote_objects
    del _remote_objects[remote_resource]


def _store(obj):
    global _remote_objects, _remote_objects_created
    remote_resource = RemoteResource(_remote_objects_created)
    _remote_objects_created += 1
    _remote_objects[remote_resource] = obj
    return remote_resource


def _get_object(obj):
    global _remote_objects
    if isinstance(obj, RemoteResourceWithPath):
        return obj.resolve_path(_remote_objects[obj.remote_resource])
    else:
        return obj


def _worker_call_function(function, args, kwargs, store):
    global _remote_objects, _remote_objects_created

    function = _get_object(function)
    args = (_get_object(v) for v in args)
    kwargs = {k: _get_object(v) for k, v in kwargs.items()}

    result = function(*args, **kwargs)
    if store:
        remote_resource = RemoteResource(_remote_objects_created)
        _remote_objects_created += 1
        _remote_objects[remote_resource] = result
        return remote_resource
    else:
        return result
