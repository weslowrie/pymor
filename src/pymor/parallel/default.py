# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import atexit
import os

from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.parallel.dummy import dummy_pool


@defaults('num_workers', 'ipython_profile', 'backend')
def new_parallel_pool(num_workers=None, ipython_profile=None, backend=None):
    """Creates a new default |WorkerPool|.

    If `backend` is not specificed, the following rules determine the backend
    to use:

        1. If `ipython_profile` is provided as an argument or set as
           a |default|, an :class:`~pymor.parallel.ipython.IPythonPool` |WorkerPool| will
           be created using the `ipcluster` script.

        2. If `num_workers` is set, a :class:`~pymor.parallel.zmq.ZMQPool` |WorkerPool|
           will be created.

        3. Otherwise, when an MPI parallel run is detected,
           an :class:`~pymor.parallel.mpi.MPIPool` |WorkerPool| will be created.

        4. Otherwise, a sequential run is assumed and
           :attr:`pymor.parallel.dummy.dummy_pool <pymor.parallel.dummy.DummyPool>`
           is returned.

    Parameters
    ----------
    num_workers
        For the `zmq` and `ipython` backends the number of worker processes
        to create. If `None` the number of workers is determindes via
        `os.sched_getaffinity`
    ipython_profile
        The IPython profile to use for the `ipython` backend.
    backend
        The |WorkerPool| implementation to use. Possible values are
        `'zmq'`, `'ipython'`, `'mpi'`, `'dummy'`.
    """

    global _pool

    from pymor.tools import mpi

    if _pool:
        logger = getLogger('pymor.parallel.default.new_parallel_pool')
        logger.warning('new_parallel_pool already called; returning old pool (this might not be what you want).')
        return _pool[1]

    if backend is None:
        if ipython_profile:
            backend = 'ipython'
        elif num_workers:
            backend = 'zmq'
        elif mpi.parallel:
            backend = 'mpi'
        else:
            backend = 'dummy'

    if num_workers is None and backend in ('zmq', 'ipython'):
        num_workers = len(os.sched_getaffinity(0))

    if backend == 'zmq':
        from pymor.parallel.zmq import new_zmq_pool
        np = new_zmq_pool(num_workers)
        pool = np.__enter__()
        _pool = ('zmq', pool, np)
        return pool
    elif backend == 'ipython':
        from pymor.parallel.ipython import new_ipcluster_pool
        np = new_ipcluster_pool(profile=ipython_profile, num_engines=num_workers)
        pool = np.__enter__()
        _pool = ('ipython', pool, np)
        return pool
    elif backend == 'mpi':
        if mpi.parallel:
            from pymor.parallel.mpi import MPIPool
            pool = MPIPool()
            _pool = ('mpi', pool)
            return pool
        else:
            _pool = ('dummy', dummy_pool)
            return dummy_pool
    elif backend == 'dummy':
        _pool = ('dummy', dummy_pool)
        return dummy_pool
    else:
        raise NotImplementedError


_pool = None


@atexit.register
def _cleanup():
    global _pool
    if _pool and _pool[0] in ('zmq', 'ipython'):
        _pool[2].__exit__(None, None, None)
    _pool = None
