# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import sys
from pymor.core.config import config
from pymor.models.basic import *

if config.HAVE_PYCLAW:
    from clawpack import pyclaw
    import numpy as np

    from pymor.core.defaults import defaults
    from pymor.core.interfaces import BasicInterface
    from pymor.operators.basic import OperatorBase
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    class PyClawOperator(OperatorBase):
        """PyClaw |Operator|"""

        linear = True

        def __init__(self, claw, source_space, range_space, solver_options=None, name=None):
            self.__auto_init(locals())
            self.source = source_space
            self.range = range_space
            self.claw = claw  # PyClaw Controller Object

        def apply(self, U, mu=None):
            """
            Solves using step_hyperbolic() and refactoring result such that it is a dq quantity and
            time-step, dt is divided out.  Step hyperbolic takes the current solution state and
            updates it as such:
              q^(n+1) = q^n - dt/dx*apdq - dt/dx*amdq - dt/dx*(F_(i) - F_(i-1))/kappa
              where apdq, amdq are the positive and negative waves from the Riemann solver,
              and F is the 2nd order correction flux into and out of the finite volume cell
            See Leveque, R.J., Finite Volume Methods for Hyperbolic Problems,
                  Cambridge Texts in Applied Mathematics, 2002. for details, or check the
            Clawpack/PyClaw documentation at: http://www.clawpack.org/
            """
            assert U in self.source

            # Get current state and solve hyperbolic step with PyClaw
            q0 = np.copy(self.claw.solution.state.q, order='F')  # Solution state before hyperbolic step
            print("t:", self.claw.solution.t, "dt:", self.claw.solver.dt)
            self.claw.solver.step_hyperbolic(self.claw.solution)  # Solve FV using PyClaw

            # Refactor result to be compatible with axpy() in time-stepping routine
            q = -(self.claw.solution.state.q - q0)/self.claw.solver.dt
            q = np.reshape(q, self.claw.solution.state.q.size, order='F')

            self.claw.solution.t += self.claw.solver.dt  # Update solution time

            # Using step()
            # self.claw.solver.step(self.claw.solution, take_one_step=True, tstart=0, tend=self.claw.solver.dt)

            # Return vector space with result
            return NumpyVectorSpace(self.claw.solution.state.q.size).make_array([q])

    class PyClawModel():

        def __init__(self, claw):
            self.claw = claw
            self.num_eqn = claw.solution.state.q.shape[0]
            self.mx = claw.solution.state.q.shape[1]
            self.my = claw.solution.state.q.shape[2]
            self.mz = claw.solution.state.q.shape[3]
            self.solution_space = NumpyVectorSpace(claw.solution.state.q.size)
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
                self.claw.frames.append(copy.deepcopy(self.claw.solution))  # initial solution
                self.claw.solution.write(frame, self.claw.outdir,
                                         self.claw.output_format,
                                         self.claw.output_file_prefix,
                                         self.claw.write_aux_always,
                                         self.claw.output_options)
                t = 0.0
                t_increment = 10.0
                for i in range(12):
                    self.claw.solver.evolve_to_time(self.claw.solution, t+t_increment)
                    frame.increment()
                    self.claw.frames.append(copy.deepcopy(self.claw.solution))
                    self.claw.solution.write(frame, self.claw.outdir,
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
                qr.append(np.reshape(self.claw.frames[f].q, dim, order='F'))
            return self.solution_space.make_array(qr)

    class PyClawVisualizer(BasicInterface):
        """Visualize a PyClaw Solution

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
            import matplotlib.pyplot as plt
            from matplotlib.colors import BoundaryNorm
            from matplotlib.ticker import MaxNLocator

            mx = self.claw.solution.state.q.shape[1]
            my = self.claw.solution.state.q.shape[2]
            mz = self.claw.solution.state.q.shape[3]

            Ur = np.reshape(U.to_numpy(), (U.to_numpy().shape[0], self.claw.solver.num_eqn, mx, my, mz), order='F')

            sys.path.append(os.getcwd())
            from mappedGrid import euler3d_mappedgrid as mg
            import hotsphere as hs

            # Atmosphere Data
            mbc = self.claw.solver.num_ghost
            print("mbc:", mbc)
            pmzBC = mz + 2 * mbc
            p0   = np.zeros([pmzBC], dtype='float', order='F')
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

            # Conserved Quantities Advanced by Solver (Euler Gas Dynamics Equations)
            rho  = Ur[index, 0, :, 79, :]  # Mass Density
            rhou = Ur[index, 1, :, 79, :]  # Momentum x-direction
            rhov = Ur[index, 2, :, 79, :]  # Momentum y-direction
            rhow = Ur[index, 3, :, 79, :]  # Momentum z-direction
            ene  = Ur[index, 4, :, 79, :]  # Energy Density

            # Compute Derived Quantities
            velsq = (rhou**2 + rhov**2 + rhow**2) / rho**2  # velocity squared
            p = (hs.gamma - 1.0) * (ene - 0.5 * rho * velsq)  # gas pressure
            T = np.zeros([np.size(p, 0), np.size(p, 1)], dtype='float', order='F')
            for i in range(np.size(p, 0)):
                for k in range(np.size(p, 1)):
                    # Compute Temperature
                    T[i, k] = p[i, k] / (hs.nAvogadro / Mavg[k + mbc] * rho[i, k]) / hs.kBoltzmann
            LogT = np.log(T)  # Log Temperature
            LogRho = np.log(rho)  # Log Mass Density
            Vmag = np.sqrt(velsq)  # Velocity Magnitude
            cSound = np.sqrt(hs.gamma * p / rho)  # Sound Speed

            plt.figure()
            plt.title(r'PyClaw Finite Volume Solution')

            plt.subplot(221)
            plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], LogRho, cmap='jet', vmin=-40, vmax=-18, shading='gouraud')
            plt.title('Log Mass Density')
            plt.colorbar()

            plt.subplot(222)
            plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], LogT, vmin=0.0, vmax=10.0, cmap='jet', shading='gouraud')
            plt.title('Log Temperature')
            plt.colorbar()

            plt.subplot(223)
            plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], Vmag, cmap='jet', vmin=0., vmax=5.e5, shading='gouraud')
            plt.title('Velocity Magnitude')
            plt.colorbar()

            plt.subplot(224)
            plt.pcolormesh(Xp[:, 79, :], Zp[:, 79, :], cSound, cmap='jet', shading='gouraud')
            plt.title('Sound Speed, c_s')
            plt.colorbar()

            plt.show()
