# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import sys
from pymor.core.config import config
from pymor.models.basic import *
#from pymor.models.interfaces import VectorArrayInterface

if config.HAVE_PYCLAW:
    from clawpack import pyclaw
    import numpy as np

    from pymor.core.defaults import defaults
    from pymor.core.interfaces import BasicInterface
    from pymor.operators.basic import OperatorBase
    from pymor.vectorarrays.interfaces import _create_random_values
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace, NumpyVector

    class PyClawVectorSpace(ListVectorSpace):

        def __init__(self, claw, id='STATE'):
            self.__auto_init(locals())
            self.claw = claw
            self.num_eqn = claw.solution.state.q.shape[0]
            self.mx = claw.solution.state.q.shape[1]
            self.my = claw.solution.state.q.shape[2]
            self.mz = claw.solution.state.q.shape[3]

        @property
        def dim(self):
            return self.claw.solution.state.q.size

        def __eq__(self, other):
            return type(other) is PyClawVectorSpace and self.claw == other.claw and self.id == other.id

        # since we implement __eq__, we also need to implement __hash__
        def __hash__(self):
            return id(self.claw) + hash(self.id)

        def zero_vector(self):
            return NumpyVector(np.zeros(self.dim))

        def full_vector(self, value):
            return NumpyVector(np.ones(self.dim)*value)

        def random_vector(self, distribution, random_state, **kwargs):
            values = _create_random_values(self.dim,distribution,random_state,**kwargs)
            return NumpyVector(np.ones(self.dim)*values)

        def make_vector(self, obj):
            return NumpyVector(obj)

    class PyClawOperator(OperatorBase):
        """PyClaw |Operator|"""

        linear = True

        def __init__(self,matrix, source_space, range_space, solver_options=None, name=None):
            assert matrix.rank() == 2
            self.__auto_init(locals())
            self.source = PyClawVectorSpace(source_space)
            self.range = PyClawVectorSpace(range_space)

        def apply(self, U, mu=None):
            assert U in self.source
            R = self.range.zeros(len(U))

            return R

    class PyClawModel:

        def __init__(self, claw):
            self.claw = claw
            self.num_eqn = claw.solution.state.q.shape[0]
            self.mx = claw.solution.state.q.shape[1]
            self.my = claw.solution.state.q.shape[2]
            self.mz = claw.solution.state.q.shape[3]
            self.solution_space = PyClawVectorSpace(claw)

        def solve(self, mu=None):
            assert mu is None
            status = self.claw.run()
            dim = self.claw.frames[0].q.size
            qr = []
            for f in range(len(self.claw.frames)):
                qr.append(np.reshape(self.claw.frames[f].q,dim,order='F'))
            return self.solution_space.make_array(qr)

        def visualize(self, U, index):
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.colors import BoundaryNorm
            from matplotlib.ticker import MaxNLocator
            import numpy as np

            Ur = np.reshape(U.to_numpy(), (U.to_numpy().shape[0], self.num_eqn, self.mx, self.my, self.mz), order='F')
            oneD = False
            twoD = True

            if oneD:
                zc = np.linspace(-1., 1., Ur.shape[4])
                plt.figure()
                plt.title(r'PyClaw Finite Volume Solution')
                plt.subplot(131)
                plt.plot(zc, Ur[index,0, 5, 5, :], 'b-', linewidth=2, label='Density')
                plt.legend()
                plt.subplot(132)
                plt.plot(zc, Ur[index,4, 5, 5, :], 'g-', linewidth=2, label='Pressure')
                plt.legend()
                plt.subplot(133)
                plt.plot(zc, Ur[index,3, 5, 5, :], 'r-', linewidth=2, label='Velocity')
                plt.legend()
                plt.show()

            if twoD:
                x = np.linspace(0.,2.0,self.mx)
                y = np.linspace(0.,0.5,self.my)
                z = np.linspace(0.,0.5,self.mz)
                X,Y = np.meshgrid(x,y,indexing='ij')
                mz = self.mz

                cmap = plt.get_cmap('gist_rainbow')
                #levels = MaxNLocator(nbins=30).tick_values(0.0,2.0)

                plt.figure()
                plt.title(r'PyClaw Finite Volume Solution')
                plt.contourf(X[:,:],Y[:,:],Ur[index,0,:,:,16])
                #plt.contourf(X[:,:],Y[:,:],Ur[index,4,:,:,mz/2],levels=levels,cmap=cmap,
                #             interpolation='bicubic',origin='upper')
                #plt.pcolormesh(Ur[index,0,:,:,16])
                plt.colorbar()
                plt.show()