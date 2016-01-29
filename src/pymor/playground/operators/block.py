# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.operators.basic import OperatorBase
from pymor.operators.interfaces import OperatorInterface
from pymor.vectorarrays.block import BlockVectorArray
from pymor.vectorarrays.interfaces import VectorSpace


class BlockOperator(OperatorBase):
    """A sparse matrix of arbitrary operators

    Parameters
    ----------
    blocks
        Two-dimensional |NumPy| array where each entry is an operator or None.
    """

    def _operators(self):
        """Iterate over operators (not None)"""
        for row in self._blocks:
            for entry in row:
                if entry is not None:
                    yield entry

    def __init__(self, blocks):
        assert isinstance(blocks, np.ndarray) and blocks.ndim == 2
        self._blocks = blocks

        assert all(isinstance(op, OperatorInterface) for op in self._operators())
        assert all(any(self._blocks[i, j] is not None for j in xrange(self._blocks.shape[1]))
                   for i in xrange(self._blocks.shape[0]))
        assert all(any(self._blocks[i, j] is not None for i in xrange(self._blocks.shape[0]))
                   for j in xrange(self._blocks.shape[1]))

        source_types = [None for j in xrange(self._blocks.shape[1])]
        range_types = [None for i in xrange(self._blocks.shape[0])]
        for (i, j), op in np.ndenumerate(self._blocks):
            if op is not None:
                assert source_types[j] is None or op.source == source_types[j]
                source_types[j] = op.source
                assert range_types[i] is None or op.range == range_types[i]
                range_types[i] = op.range

        self.source = VectorSpace(BlockVectorArray, tuple(source_types))
        self.range = VectorSpace(BlockVectorArray, tuple(range_types))
        self._source_dims = tuple(space.dim for space in self.source.subtype)
        self._range_dims = tuple(space.dim for space in self.range.subtype)
        self.num_source_blocks = len(source_types)
        self.num_range_blocks = len(range_types)
        self.linear = all(op.linear for op in self._operators())
        self._is_diagonal = (all(block is None if i != j else True for (i, j), block in np.ndenumerate(self._blocks))
                             and self.num_source_blocks == self.num_range_blocks)
        self.build_parameter_type(inherits=list(self._operators()))

    @classmethod
    def hstack(cls, operators):
        """Horizontal stacking of operators

        Parameters
        ----------
        operators
            A tuple, list, array, or iterable of operators.
        """
        blocks = np.array([[op for op in operators]])
        return cls(blocks)

    @classmethod
    def vstack(cls, operators):
        """Vertical stacking of operators

        Parameters
        ----------
        operators
            A tuple, list, array, or iterable of operators.
        """
        blocks = np.array([[op] for op in operators])
        return cls(blocks)

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        assert U.check_ind(ind)

        blocks = []
        for i in xrange(self.num_range_blocks):
            block = None
            for j in xrange(self.num_source_blocks):
                op = self._blocks[i, j]
                if op is not None:
                    V = op.apply(U.block(j))
                    if block is None:
                        block = V.copy()
                    else:
                        block += V
            blocks.append(block.copy())

        return BlockVectorArray(blocks)
