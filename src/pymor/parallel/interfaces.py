# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import BasicInterface, abstractmethod


class WorkerPoolInterface(BasicInterface):
    """Interface for parallel worker pools.

    |WorkerPools| allow to easily parallelize algorithms which involve
    no or little communication between the workers at runtime. The interface
    methods give the user simple means to distribute data to
    workers (:meth:`~WorkerPoolInterface.push`, :meth:`~WorkerPoolInterface.scatter_array`,
    :meth:`~WorkerPoolInterface.scatter_list`) and execute functions on
    the distributed data in parallel (:meth:`~WorkerPoolInterface.apply`),
    collecting the return values from each function call. A
    single worker can be instructed to execute a function using the
    :meth:`WorkerPoolInterface.apply_only` method. Finally, a parallelized
    :meth:`~WorkerPoolInterface.map` function is available, which
    automatically scatters the data among the workers.

    All operations are performed synchronously.

    Attributes
    ----------
    remote_objects
        Set of all objects managed by the workers.
    """

    remote_objects = None

    @abstractmethod
    def __len__(self):
        """The number of workers in the pool."""
        pass

    @abstractmethod
    def push(self, obj):
        """Push a copy of `obj` to  all workers of the pool.

        A |RemoteObject| is returned as a handle to the pushed object.
        This object can be used as an argument to :meth:`~WorkerPoolInterface.apply`,
        :meth:`~WorkerPoolInterface.map` and will then be transparently mapped to the
        respective copy of the pushed object on the worker.

        |Immutable| objects will be pushed only once. If the same |immutable| object
        is pushed a second time, the returned |RemoteObject| will refer to the
        already transferred copy. It is therefore safe to use `push` to ensure
        that a given |immutable| object is available on the worker. No unnecessary
        copies will be created.

        Parameters
        ----------
        obj
            The object to push to all workers.

        Returns
        -------
        A |RemoteObject| referring to the pushed data.
        """
        pass

    @abstractmethod
    def scatter(self, l):
        """Distribute a |VectorArray| or a list of objects evenly among the workers.

        On each worker a |VectorArray| or `list` is created holding an (up to rounding)
        equal amount of objects of `l`. The returned |RemoteObject| therefore refers
        to different data on each of the workers.

        If `len(l)` is equal to the size of the pool, it is guaranteed that the n-th
        element of `l` is assigned to the n-th worker of the pool.

        Parameters
        ----------
        l
            The list (sequence) of objects to distribute.

        Returns
        -------
        A |RemoteObject| referring to the scattered data.
        """
        pass

    @abstractmethod
    def get(self, obj, worker=None):
        """Communicate object refered to by the given |RemoteObject|

        Parameters
        ----------
        obj
            |RemoteObject| or |RemotePath| to communicate.
        worker
            If not `None`, list of workers for which to communicate `obj`.

        Returns
        -------
        List of communicated objects of length `len(worker)`.
        """
        pass

    @abstractmethod
    def apply(self, function, *args, store=False, worker=None, **kwargs):
        """Apply function in parallel on each worker.

        This calls `function` on each worker in parallel, passing `args` as
        positional and `kwargs` as keyword arguments. Positional or keyword
        arguments which are |RemoteObjects| or |RemotePaths| are automatically
        mapped to the respective object on the worker. Moreover, arguments
        which are |immutable| objects that have already been pushed to the workers
        will not be transmitted again. (|Immutable| objects which have not
        been pushed before will be transmitted and the remote copy will be
        destroyed after function execution.)

        Parameters
        ----------
        function
            The function to execute on each worker.
        args
            The positional arguments for `function`.
        kwargs
            The keyword arguments for `function`.
        store
            If `True`, do not communicate the results but return a
            |RemoteObject| instead. Only possible when `worker` is
            `None`.
        worker
            If not `None`, list of workers for which to execute `function`.

        Returns
        -------
        List of return values of the function executions, ordered by
        worker number (from `0` to `len(pool) - 1`), in case `store`
        is `True`. Otherwise a |RemoteObject| referring to the results.
        """

    @abstractmethod
    def map(self, function, *args, **kwargs):
        """Parallel version of the builtin :func:`map` function.

        Each positional argument (after `function`) must be a sequence
        of same length n. `map` calls `function` in parallel on each of these n
        positional argument combinations, always passing `kwargs` as keyword
        arguments.  Keyword arguments which are |RemoteObjects| or |RemotePaths|
        are automatically mapped to the respective object on the worker. Moreover,
        keyword arguments which are |immutable| objects that have already been
        pushed to the workers will not be transmitted again. (|Immutable| objects
        which have not been pushed before will be transmitted and the remote copy
        will be destroyed after function execution.)

        Parameters
        ----------
        function
            The function to execute on each worker.
        args
            The sequences of positional arguments for `function`.
        kwargs
            The keyword arguments for `function`.

        Returns
        -------
        List of return values of the function executions, ordered by
        the sequence of positional arguments.
        """
        pass


class RemoteObjectBase:
    pool = None

    def __call__(self, *args, worker=None, store=False, **kwargs):
        return self.pool.apply(self, *args, worker=worker, store=store, **kwargs)

    def get(self, worker=None):
        return self.pool.get(self, worker=worker)


class RemoteObject(RemoteObjectBase):
    """Handle to remote data on the workers of a |WorkerPool|.

    See documentation of :class:`WorkerPoolInterface` for usage
    of these handles in conjunction with :meth:`~WorkerPoolInterface.apply`,
    :meth:`~WorkerPoolInterface.scatter` and
    :meth:`~WorkerPoolInterface.map`.

    Accessing an arbitrary attribute of an RemoteObject or indexing
    with an arbitrary index yields a |RemotePath| holding a symbolic
    reference to the respective object.
    """

    def __init__(self, pool, remote_resource):
        self.pool, self.remote_resource = pool, remote_resource

    def __getattr__(self, name):
        return RemotePath(self, [name])

    def __getitem__(self, key):
        return RemotePath(self, [[key]])


class RemotePath(RemoteObjectBase):
    """|RemoteObject| together with a path of attributes/indices."""

    def __init__(self, remote_object, path):
        self.remote_object, self.path = remote_object, path
        self.pool = remote_object.pool

    def __getattr__(self, name):
        return RemotePath(self.remote_object, self.path + [name])

    def __getitem__(self, key):
        return RemotePath(self.remote_object, self.path + [[key]])
