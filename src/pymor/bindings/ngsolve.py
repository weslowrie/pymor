# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
from pymor.core.defaults import defaults

if config.HAVE_NGSOLVE:
    import ngsolve as ngs
    import numpy as np

    from pymor.core.interfaces import ImmutableInterface
    from pymor.operators.basic import OperatorBase
    from pymor.operators.constructions import ZeroOperator
    from pymor.vectorarrays.interfaces import VectorArrayInterface
    from pymor.vectorarrays.numpy import NumpyVectorSpace
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace


    class NGSolveVector(CopyOnWriteVector):
        """Wraps a NGSolve BaseVector to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl)

        def _copy_data(self):
            new_impl = ngs.GridFunction(self.impl.space)
            new_impl.vec.data = self.impl.vec
            self.impl = new_impl

        @property
        def data(self):
            return self.impl.vec.FV().NumPy()

        def _scal(self, alpha):
            self.impl.vec.data = float(alpha) * self.impl.vec

        def _axpy(self, alpha, x):
            self.impl.vec.data = self.impl.vec + float(alpha) * x.impl.vec

        def dot(self, other):
            return self.impl.vec.InnerProduct(other.impl.vec)

        def l1_norm(self):
            return np.linalg.norm(self.data, ord=1)

        def l2_norm(self):
            return self.impl.vec.Norm()

        def l2_norm2(self):
            return self.impl.vec.Norm() ** 2

        def dofs(self, dof_indices):
            return self.data[dof_indices]

        def amax(self):
            A = np.abs(self.data)
            max_ind = np.argmax(A)
            max_val = A[max_ind]
            return max_ind, max_val


    class NGSolveVectorSpace(ListVectorSpace):

        def __init__(self, V, id_='STATE'):
            self.V = V
            self.id = id_

        def __eq__(self, other):
            return type(other) is NGSolveVectorSpace and self.V == other.V and self.id == other.id

        def __hash__(self):
            return hash(self.V) + hash(self.id)

        @property
        def value_dim(self):
            u = self.V.TrialFunction()
            if isinstance(u, list):
                return u[0].dim
            else:
                return u.dim

        @property
        def dim(self):
            return self.V.ndofglobal * self.value_dim

        @classmethod
        def space_from_vector_obj(cls, vec, id_):
            return cls(vec.space, id_)

        def zero_vector(self):
            impl = ngs.GridFunction(self.V)
            return NGSolveVector(impl)

        def make_vector(self, obj):
            return NGSolveVector(obj)

        def vector_from_data(self, data):
            v = self.zero_vector()
            v.data[:] = data
            return v


    class NGSolveMatrixOperator(OperatorBase):
        """Wraps a NGSolve matrix as an |Operator|."""

        linear = True

        def __init__(self, matrix, range, source, solver_options=None, free_dofs=None, name=None):
            self.range = range
            self.source = source
            self.matrix = matrix
            self.solver_options = solver_options
            self.free_dofs = free_dofs
            self.name = name

        def apply(self, U, mu=None):
            assert U in self.source
            R = self.range.zeros(len(U))
            for u, r in zip(U._list, R._list):
                self.matrix.Mult(u.impl.vec, r.impl.vec)
            return R

        def apply_transpose(self, V, mu=None):
            assert V in self.range
            U = self.source.zeros(len(V))
            mat = self.matrix.Transpose()
            for v, u in zip(V._list, U._list):
                mat.Mult(v.impl.vec, u.impl.vec)
            return U

        @defaults('default_solver')
        def apply_inverse(self, V, mu=None, least_squares=False, default_solver=''):
            assert V in self.range
            if least_squares:
                raise NotImplementedError
            solver = self.solver_options.get('inverse', default_solver) if self.solver_options else default_solver
            R = self.source.zeros(len(V))
            with ngs.TaskManager():
                inv = self.matrix.Inverse(self.free_dofs or self.source.V.FreeDofs(), inverse=solver)
                for r, v in zip(R._list, V._list):
                    r.impl.vec.data = inv * v.impl.vec
            return R

        def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
            if not all(isinstance(op, (NGSolveMatrixOperator, ZeroOperator)) for op in operators):
                return None

            matrix = operators[0].matrix.CreateMatrix()
            matrix.AsVector().data = float(coefficients[0]) * matrix.AsVector()
            for op, c in zip(operators[1:], coefficients[1:]):
                if isinstance(op, ZeroOperator):
                    continue
                matrix.AsVector().data += float(c) * op.matrix.AsVector()
            return NGSolveMatrixOperator(matrix, self.range, self.source, solver_options=solver_options, free_dofs=operators[0].free_dofs, name=name)

        def as_vector(self, copy=True):
            vec = self.matrix.AsVector().FV().NumPy()
            return NumpyVectorSpace.make_array(vec.copy() if copy else vec)



    class NGSolveVisualizer(ImmutableInterface):
        """Visualize an NGSolve grid function."""

        def __init__(self, mesh, fespace):
            self.mesh = mesh
            self.fespace = fespace
            self.space = NGSolveVectorSpace(fespace)

        def visualize(self, U, d, legend=None, separate_colorbars=True, block=True):
            """Visualize the provided data."""
            if isinstance(U, VectorArrayInterface):
                U = (U,)
            assert all(u in self.space for u in U)
            if any(len(u) != 1 for u in U):
                raise NotImplementedError

            if legend is None:
                legend = ['VectorArray{}'.format(i) for i in range(len(U))]
            if isinstance(legend, str):
                legend = [legend]
            assert len(legend) == len(U)
            legend = [l.replace(' ', '_') for l in legend]  # NGSolve GUI will fail otherwise

            if not separate_colorbars:
                raise NotImplementedError

            for u, name in zip(U, legend):
                ngs.Draw(u._list[0].impl, self.mesh, name=name)

    def visualize_arrays_vtk(arrays, filename, names=None, subdivision=0):
        mesh = arrays[0].space.V.mesh
        assert all(U.space.V.mesh == mesh for U in arrays)
        if not names:
            names = ['U{}'.format(i) for i in range(len(arrays))]
        assert len(arrays) == len(names)
        assert all(len(U) == len(arrays[0]) for U in arrays)

        gfs = [U.zeros()._list[0].impl for U in arrays]
        all_gfs = []
        all_names = []
        for gf, name in zip(gfs, names):
            if gf.components:  # multi-component GridFunctions do not seem to be handled correctly by NGSolve ATM
                all_gfs.extend(gf.components)
                all_names.extend(['{}_{}'.format(name, i) for i in range(len(gf.components))])
            else:
                all_gfs.append(gf)
                all_names.append(name)

        vtk = ngs.VTKOutput(ma=mesh, coefs=all_gfs, names=all_names, filename=filename, subdivision=subdivision)

        for i in range(len(arrays[0])):
            gf_iter = iter(all_gfs)
            for U in arrays:
                gf_data = U._list[i].impl
                if gf_data.components:
                    for c in gf_data.components:
                        next(gf_iter).vec.data = c.vec
                else:
                    next(gf_iter).vec.data = gf_data.vec
            vtk.Do()

        del all_gfs
