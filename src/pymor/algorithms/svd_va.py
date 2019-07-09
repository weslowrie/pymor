# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.logger import getLogger
from pymor.operators.interfaces import OperatorInterface
from pymor.vectorarrays.interfaces import VectorArrayInterface


def qr_svd(A, product=None):
    """SVD of a |VectorArray|.

    If `product` is given, left singular vectors will be orthogonal with
    respect to it. Otherwise, the Euclidean inner product is used.

    Parameters
    ----------
    A
        The |VectorArray| for which the SVD is to be computed.
        The vectors are interpreted as columns in a matrix.
    product
        Inner product |Operator| w.r.t. which the left singular vectors are
        computed.

    Returns
    -------
    U
        |VectorArray| of left singular vectors.
    s
        Sequence of singular values.
    Vh
        |NumPy array| of right singular vectors.
    """

    assert isinstance(A, VectorArrayInterface)
    assert len(A) > 0
    assert product is None or isinstance(product, OperatorInterface)

    logger = getLogger('pymor.algorithms.qr_svd.qr_svd')

    with logger.block('Computing QR decomposition ...'):
        Q, R = gram_schmidt(A, product=product, return_R=True, atol=0, rtol=0)

    logger.info('Computing SVD of R.')
    U2, s, Vh = spla.svd(R, lapack_driver='gesvd')

    logger.info('Computing left singular vectors.')
    U = Q.lincomb(U2.T)

    return U, s, Vh
