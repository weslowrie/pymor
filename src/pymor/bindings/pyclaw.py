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
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    """
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
    """

    class PyClawOperator(OperatorBase):
        """PyClaw |Operator|"""

        linear = True

        def __init__(self, claw, source_space, range_space, solver_options=None, name=None):
            self.__auto_init(locals())
            self.source = source_space
            self.range = range_space
            self.claw = claw # PyClaw Controller Object


        def apply(self, U, mu=None):
            """
            Solves using step_hyperbolic() and refactoring result such that it is a dq quantity and
            timestep, dt is divided out.  Step hyperbolic takes the current solution state and updates
            it as such:
              q^(n+1) = q^n - dt/dx*apdq - dt/dx*amdq - dt/dx*(F_(i) - F_(i-1))/kappa
              where apdq, amdq are the positive and negative waves from the Riemann solver,
              and F is the 2nd order correction flux into and out of the finite volume cell
            See Leveque, R.J., Finite Volume Methods for Hyperbolic Problems,
                  Cambridge Texts in Applied Mathematics, 2002. for details, or check the
            Clawpack/PyClaw documentation at: http://www.clawpack.org/
            """
            assert U in self.source

            # Get current state and solve hyperbolic step with PyClaw
            state = self.claw.solution.states[0] # Get the current solution state
            q0 = np.copy(self.claw.solution.state.q, order='F') # Solution state before hyperbolic step
            print("t:", self.claw.solution.t, "dt:", self.claw.solver.dt)
            self.claw.solver.step_hyperbolic(self.claw.solution) # Solve FV using PyClaw

            # Refactor result to be compatible with axpy() in timestepping routine
            q = -(self.claw.solution.state.q - q0)/self.claw.solver.dt
            q = np.reshape(q, self.claw.solution.state.q.size, order='F')

            self.claw.solution.t += self.claw.solver.dt # Update solution time

            # Return vector space with result
            R = NumpyVectorSpace(self.claw.solution.state.q.size).make_array([q])

            # Using step()
            # self.claw.solver.step(self.claw.solution, take_one_step=True, tstart=0, tend=self.claw.solver.dt)

            return R

    class PyClawModel():

        def __init__(self, claw):
            self.claw = claw
            self.num_eqn = claw.solution.state.q.shape[0]
            self.mx = claw.solution.state.q.shape[1]
            self.my = claw.solution.state.q.shape[2]
            self.mz = claw.solution.state.q.shape[3]
            self.solution_space = NumpyVectorSpace(claw.solution.state.q.size) #PyClawVectorSpace(claw)
            self.claw.start_frame = self.claw.solution.start_frame

        def solve_manual(self, mu=None):
            assert mu is None
            use_claw_run = True
            if use_claw_run:
                status = self.claw.run()
            else:
                import copy
                from clawpack.pyclaw.util import FrameCounter
                frame = FrameCounter()
                frame.set_counter(self.claw.start_frame)
                self.claw.frames.append(copy.deepcopy(self.claw.solution)) # initial solution
                self.claw.solution.write(frame,self.claw.outdir,
                                        self.claw.output_format,
                                        self.claw.output_file_prefix,
                                        self.claw.write_aux_always,
                                        self.claw.output_options)
                t = 0.0
                t_increment = 10.0
                for i in range(12):
                    self.claw.solver.evolve_to_time(self.claw.solution,t+t_increment)
                    frame.increment()
                    self.claw.frames.append(copy.deepcopy(self.claw.solution))
                    self.claw.solution.write(frame,self.claw.outdir,
                                             self.claw.output_format,
                                             self.claw.output_file_prefix,
                                             self.claw.write_aux_always,
                                             self.claw.output_options)
                    self.claw.solution._start_frame = len(self.claw.frames)
                    t += t_increment

            dim = self.claw.frames[0].q.size
            print(dim)
            qr = []
            for f in range(len(self.claw.frames)):
                qr.append(np.reshape(self.claw.frames[f].q,dim,order='F'))
            return self.solution_space.make_array(qr)

        #def visualize(self, U, index):


    class PyClawVisualizer(BasicInterface):
        """Visualize a PyClaw Space

        Parameters
        ----------
        claw
            The PyClaw controller object
        """

        def __init__(self, claw):
            self.claw = claw

        def visualize(self, U, title='', legend=None, filename=None, block=True,
                      separate_colorbars=True):
            import os
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.colors import BoundaryNorm
            from matplotlib.ticker import MaxNLocator
            import numpy as np

            mx = self.claw.solution.state.q.shape[1]
            my = self.claw.solution.state.q.shape[2]
            mz = self.claw.solution.state.q.shape[3]

            Ur = np.reshape(U.to_numpy(), (U.to_numpy().shape[0], self.claw.solver.num_eqn, mx, my, mz), order='F')

            sys.path.append(os.getcwd())
            from mappedGrid import euler3d_mappedgrid as mg
            import hotsphere as hs

            # Atmosphere Data
            mbc = 2
            pmzBC = mz + 2 * mbc
            p0 = np.zeros([pmzBC], dtype='float', order='F')
            rho0 = np.zeros([pmzBC], dtype='float', order='F')
            Mavg = np.zeros([pmzBC], dtype='float', order='F')
            p0, rho0, Mavg = hs.setEquilibriumAtmosphere(p0, rho0, Mavg)

            # Create Grid
            xc, yc, zc = np.linspace(0.0, 1.0, mx), np.linspace(0.0, 1.0, my), np.linspace(0.0, 1.0, mz)
            Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')
            Xp, Yp, Zp = mg.mapc2pwrapper(Xc, Yc, Zc, mz, hs.xyzMin, hs.xyzMax, hs.mapType)
            Xp, Yp, Zp = np.reshape(Xp, [mx, my, mz], order='F'), \
                         np.reshape(Yp, [mx, my, mz], order='F'), \
                         np.reshape(Zp, [mx, my, mz], order='F')

            print("Ur:",np.shape(Ur))
            index = np.size(Ur,0)-1

            # Compute Quantities
            rho = Ur[index, 0, :, 79, :]
            velsq = (Ur[index, 1, :, 79, :] ** 2 + Ur[index, 2, :, 79, :] ** 2 + Ur[index, 3, :, 79,
                                                                                 :] ** 2) / rho ** 2
            p = (hs.gamma - 1.0) * (Ur[index, 4, :, 79, :] - 0.5 * rho * velsq)
            T = np.zeros([np.size(p, 0), np.size(p, 1)], dtype='float', order='F')
            for i in range(np.size(p, 0)):
                for k in range(np.size(p, 1)):
                    T[i, k] = p[i, k] / (hs.nAvogadro / Mavg[k + mbc] * rho[i, k]) / hs.kBoltzmann
            LogT = np.log(T)
            print("rho:",np.shape(rho),np.min(rho),np.max(rho))
            LogRho = np.log(rho)
            Vmag = np.sqrt(velsq)
            cSound = np.sqrt(1.4 * p / rho)

            plt.figure()
            plt.title(r'PyClaw Finite Volume Solution')

            plt.subplot(221)
            # plt.contourf(Xp[:,79,:], Zp[:,79,:],LogRho,np.arange(-40.,-18.,2.2),extend='both',cmap='jet')
            plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], LogRho, cmap='jet', vmin=-40, vmax=-18, shading='gouraud')
            plt.title('Log Mass Density')
            plt.colorbar()

            plt.subplot(222)
            # plt.contourf(Xp[:,79,:], Zp[:,79,:],LogT,np.arange(4.0,10.0,0.5),extend='both',cmap='jet')
            plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], LogT, vmin=0.0, vmax=10.0, cmap='jet', shading='gouraud')
            plt.title('Log Temperature')
            plt.colorbar()

            plt.subplot(223)
            # plt.contourf(Xp[:,79,:], Zp[:,79,:],Vmag,np.arange(0.0,5.e5,5.e4),extend='both',cmap='jet')
            plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], Vmag, cmap='jet', vmin=0., vmax=5.e5, shading='gouraud')
            plt.title('Velocity Magnitude')
            plt.colorbar()

            plt.subplot(224)
            # plt.contourf(Xp[:,79,:], Zp[:,79,:],cSound,extend='both',cmap='jet')
            plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], cSound, cmap='jet', shading='gouraud')
            plt.title('Sound Speed, c_s')
            plt.colorbar()

            plt.show()

        def visualize_old(self, U, index, title='', legend=None, filename=None, block=True,
                      separate_colorbars=True):
            """Visualize the provided data.

            Parameters
            ----------
            U
                |VectorArray| of the data to visualize (length must be 1). Alternatively,
                a tuple of |VectorArrays| which will be visualized in separate windows.
                If `filename` is specified, only one |VectorArray| may be provided which,
                however, is allowed to contain multipled vectors that will be interpreted
                as a time series.
            index
                Time slice index
            m
                Filled in by :meth:`pymor.models.ModelBase.visualize` (ignored).
            title
                Title of the plot.
            legend
                Description of the data that is plotted. If `U` is a tuple of |VectorArrays|,
                `legend` has to be a tuple of the same length.
            filename
                If specified, write the data to that file. `filename` needs to have an extension
                supported by FEniCS (e.g. `.pvd`).
            separate_colorbars
                If `True`, use separate colorbars for each subplot.
            block
                If `True`, block execution until the plot window is closed.
            """

            import os
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.colors import BoundaryNorm
            from matplotlib.ticker import MaxNLocator
            import numpy as np

            mx = self.claw.solution.state.q.shape[1]
            my = self.claw.solution.state.q.shape[2]
            mz = self.claw.solution.state.q.shape[3]

            Ur = np.reshape(U.to_numpy(), (U.to_numpy().shape[0], self.claw.solver.num_eqn, mx, my, mz), order='F')
            oneD = False
            twoD = False
            threeD = True

            if oneD:
                zc = np.linspace(-1., 1., Ur.shape[4])
                plt.figure()
                plt.title(r'PyClaw Finite Volume Solution')
                plt.subplot(131)
                plt.plot(zc, Ur[index, 0, 5, 5, :], 'b-', linewidth=2, label='Density')
                plt.legend()
                plt.subplot(132)
                plt.plot(zc, Ur[index, 4, 5, 5, :], 'g-', linewidth=2, label='Pressure')
                plt.legend()
                plt.subplot(133)
                plt.plot(zc, Ur[index, 3, 5, 5, :], 'r-', linewidth=2, label='Velocity')
                plt.legend()
                plt.show()

            if twoD:
                x = np.linspace(0., 2.0, mx)
                y = np.linspace(0., 0.5, my)
                z = np.linspace(0., 0.5, mz)
                X, Y = np.meshgrid(x, y, indexing='ij')

                cmap = plt.get_cmap('gist_rainbow')
                # levels = MaxNLocator(nbins=30).tick_values(0.0,2.0)

                plt.figure()
                plt.title(r'PyClaw Finite Volume Solution')
                plt.contourf(X[:, :], Y[:, :], Ur[index, 0, :, :, 16])
                # plt.contourf(X[:,:],Y[:,:],Ur[index,4,:,:,mz/2],levels=levels,cmap=cmap,
                #             interpolation='bicubic',origin='upper')
                # plt.pcolormesh(Ur[index,0,:,:,16])
                plt.colorbar()
                plt.show()

            if threeD:
                sys.path.append(os.getcwd())
                from mappedGrid import euler3d_mappedgrid as mg
                import hotsphere as hs

                # Atmosphere Data
                mbc = 2
                pmzBC = mz + 2 * mbc
                p0 = np.zeros([pmzBC], dtype='float', order='F')
                rho0 = np.zeros([pmzBC], dtype='float', order='F')
                Mavg = np.zeros([pmzBC], dtype='float', order='F')
                p0, rho0, Mavg = hs.setEquilibriumAtmosphere(p0, rho0, Mavg)

                # Create Grid
                xc, yc, zc = np.linspace(0.0, 1.0, mx), np.linspace(0.0, 1.0, my), np.linspace(0.0, 1.0, mz)
                Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')
                Xp, Yp, Zp = mg.mapc2pwrapper(Xc, Yc, Zc, mz, hs.xyzMin, hs.xyzMax, hs.mapType)
                Xp, Yp, Zp = np.reshape(Xp, [mx, my, mz], order='F'), \
                             np.reshape(Yp, [mx, my, mz], order='F'), \
                             np.reshape(Zp, [mx, my, mz], order='F')

                # Compute Quantities
                rho = Ur[index, 0, :, 79, :]
                velsq = (Ur[index, 1, :, 79, :] ** 2 + Ur[index, 2, :, 79, :] ** 2 + Ur[index, 3, :, 79,
                                                                                     :] ** 2) / rho ** 2
                p = (hs.gamma - 1.0) * (Ur[index, 4, :, 79, :] - 0.5 * rho * velsq)
                T = np.zeros([np.size(p, 0), np.size(p, 1)], dtype='float', order='F')
                for i in range(np.size(p, 0)):
                    for k in range(np.size(p, 1)):
                        T[i, k] = p[i, k] / (hs.nAvogadro / Mavg[k + mbc] * rho[i, k]) / hs.kBoltzmann
                LogT = np.log(T)
                LogRho = np.log(rho)
                Vmag = np.sqrt(velsq)
                cSound = np.sqrt(1.4 * p / rho)

                plt.figure()
                plt.title(r'PyClaw Finite Volume Solution')

                plt.subplot(221)
                # plt.contourf(Xp[:,79,:], Zp[:,79,:],LogRho,np.arange(-40.,-18.,2.2),extend='both',cmap='jet')
                plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], LogRho, cmap='jet', vmin=-40, vmax=-18, shading='gouraud')
                plt.title('Log Mass Density')
                plt.colorbar()

                plt.subplot(222)
                # plt.contourf(Xp[:,79,:], Zp[:,79,:],LogT,np.arange(4.0,10.0,0.5),extend='both',cmap='jet')
                plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], LogT, vmin=0.0, vmax=10.0, cmap='jet', shading='gouraud')
                plt.title('Log Temperature')
                plt.colorbar()

                plt.subplot(223)
                # plt.contourf(Xp[:,79,:], Zp[:,79,:],Vmag,np.arange(0.0,5.e5,5.e4),extend='both',cmap='jet')
                plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], Vmag, cmap='jet', vmin=0., vmax=5.e5, shading='gouraud')
                plt.title('Velocity Magnitude')
                plt.colorbar()

                plt.subplot(224)
                # plt.contourf(Xp[:,79,:], Zp[:,79,:],cSound,extend='both',cmap='jet')
                plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], cSound, cmap='jet', shading='gouraud')
                plt.title('Sound Speed, c_s')
                plt.colorbar()

                plt.show()