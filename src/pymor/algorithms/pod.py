# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.svd_va import mos, qr_svd
from pymor.core.defaults import defaults
from pymor.core.exceptions import AccuracyError
from pymor.core.logger import getLogger
from pymor.operators.interfaces import OperatorInterface
from pymor.tools.floatcmp import float_cmp_all
from pymor.vectorarrays.interfaces import VectorArrayInterface


@defaults('rtol', 'atol', 'l2_err', 'orthonormalize', 'check', 'check_tol')
def pod(A, modes=None, product=None, rtol=4e-8, atol=0., l2_err=0., orthonormalize=None,
        check=True, check_tol=1e-10, method='mos'):
    """Proper orthogonal decomposition of `A`.

    Viewing the |VectorArray| `A` as a `A.dim` x `len(A)` matrix,
    the return value of this method is the |VectorArray| of left-singular
    vectors of the singular value decomposition of `A`, where the inner product
    on R^(`dim(A)`) is given by `product` and the inner product on R^(`len(A)`)
    is the Euclidean inner product.

    Parameters
    ----------
    A
        The |VectorArray| for which the POD is to be computed.
    modes
        If not `None`, only the first `modes` POD modes (singular vectors) are
        returned.
    product
        Inner product |Operator| w.r.t. which the POD is computed.
    rtol
        Singular values smaller than this value multiplied by the largest singular
        value are ignored.
    atol
        Singular values smaller than this value are ignored.
    l2_err
        Do not return more modes than needed to bound the l2-approximation
        error by this value. I.e. the number of returned modes is at most ::

            argmin_N { sum_{n=N+1}^{infty} s_n^2 <= l2_err^2 }

        where `s_n` denotes the n-th singular value.
    orthonormalize
        If `True`, orthonormalize the computed POD modes again using
        the :func:`~pymor.algorithms.gram_schmidt.gram_schmidt` algorithm.
        If `None`, orthonormalize if `method == 'mos'`.
        If `False`, do not orthonormalize.
    check
        If `True`, check the computed POD modes for orthonormality.
    check_tol
        Tolerance for the orthonormality check.
    method
        Which SVD method from :mod:`~pymor.algorithms.svd_va` to use (`'mos'` or
        `'qr_svd'`).

    Returns
    -------
    POD
        |VectorArray| of POD modes.
    SVALS
        Sequence of singular values.
    """

    assert isinstance(A, VectorArrayInterface)
    assert len(A) > 0
    assert modes is None or modes <= len(A)
    assert product is None or isinstance(product, OperatorInterface)
    assert method in ('mos', 'qr_svd')

    logger = getLogger('pymor.algorithms.pod.pod')

    svd_va = mos if method == 'mos' else qr_svd
    POD, SVALS, _ = svd_va(A, product=product, modes=modes, rtol=rtol, atol=atol,
                           l2_err=l2_err, check=False)

    if orthonormalize is None:
        orthonormalize = True if method == 'mos' else False
    if orthonormalize:
        with logger.block('Re-orthonormalizing POD modes ...'):
            POD = gram_schmidt(POD, product=product, copy=False)

    if check:
        logger.info('Checking orthonormality ...')
        if not float_cmp_all(POD.inner(POD, product), np.eye(len(POD)), atol=check_tol, rtol=0.):
            err = np.max(np.abs(POD.inner(POD, product) - np.eye(len(POD))))
            raise AccuracyError(f'result not orthogonal (max err={err})')
        if len(POD) < len(SVALS):
            raise AccuracyError('additional orthonormalization removed basis vectors')

    return POD, SVALS
