#!/usr/bin/env python
# encoding: utf-8

""" 
Test problem demonstrating a 1D shocktube in a 3D domain. 

This script evolves the 3D Euler equations.
The primary variables are: 
    density (rho), x,y, and z momentum (rho*u,rho*v,rho*w), and energy.
"""
from __future__ import absolute_import
from clawpack import riemann
from clawpack.riemann.euler_3D_constants import density, x_momentum, \
                y_momentum, z_momentum, energy, num_eqn
from pymor.bindings.pyclaw import *

gamma = 1.4 # Ratio of Specific Heats

def shocktube(kernel_language='Fortran', solver_type='classic',
              use_petsc=False, outdir='shocktube_output', output_format='hdf5',
              disable_output=False, mx=10, my=10, mz=128, tfinal=1.0,
              num_output_times=10):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw
        
    if solver_type == 'classic':
        solver = pyclaw.ClawSolver3D(riemann.euler_3D)
        solver.dimensional_split = True
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.cfl_max = 1.0
        solver.cfl_desired = 0.80
    elif solver_type == 'sharpclaw':
        solver = pyclaw.SharpClawSolver3D(riemann.euler_3D)
    else:
        raise Exception('Unrecognized solver_type.')

    domain = pyclaw.Domain((-1.,-1.,-1.),(1.,1.,1.),(mx,my,mz))

    state = pyclaw.State(domain,num_eqn)
    state.problem_data['gamma']  = gamma
    
    grid = state.grid
    
    X,Y,Z = grid.p_centers

    pressure = 3.*(Z<=0) + 1.*(Z>0)
    state.q[density   ,:,:,:] = 3.*(Z<=0) + 1.*(Z>0)
    state.q[x_momentum,:,:,:] = 0.
    state.q[y_momentum,:,:,:] = 0.
    state.q[z_momentum,:,:,:] = 0.
    state.q[energy    ,:,:,:] = pressure/(gamma - 1.)

    solver.all_bcs = pyclaw.BC.extrap

    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.output_format = output_format
    claw.keep_copy = True
    if disable_output:
        claw.output_format = None
    claw.tfinal = tfinal
    claw.num_output_times = num_output_times
    claw.outdir = outdir

    return PyClawModel(claw)

# __main__()
if __name__=="__main__":
    from pymor.algorithms.pod import pod
    import matplotlib.pyplot as plt
    fom = shocktube(kernel_language='Fortran', solver_type='classic',
              use_petsc=False, outdir='shocktube_output', output_format='hdf5',
              disable_output=False, mx=10, my=10, mz=128, tfinal=0.5,
              num_output_times=100)
    U = fom.solve()
    num_output_times = len(fom.claw.frames)
    modes, svals = pod(U)

    #"""
    # plot the singualar values
    plt.figure()
    plt.semilogy(svals, linewidth=2, label='Singular Values')
    plt.legend()
    plt.show()
    #"""

    fom.visualize(U,num_output_times-1) # plot the solution
    #fom.visualize(modes,0) # plot some modes

    from pymor.algorithms.basic import *
    Up = project_array(U,modes[0:3])
    fom.visualize(Up,num_output_times-1)

    Up = project_array(U, modes[0:11])
    fom.visualize(Up, num_output_times - 1)

    Up = project_array(U, modes[0:21])
    fom.visualize(Up, num_output_times - 1)

    Up = project_array(U, modes[0:51])
    fom.visualize(Up, num_output_times - 1)

    Up = project_array(U, modes[0:101])
    fom.visualize(Up, num_output_times - 1)

    """
    index = num_output_times-1
    zc = np.linspace(-1., 1., Um.shape[4])
    plt.figure()
    plt.title(r'Mode Projection')
    plt.subplot(131)
    plt.plot(zc, Um[index, 0, 5, 5, :], 'b-', linewidth=2, label='Density')
    plt.legend()
    plt.subplot(132)
    plt.plot(zc, Um[index, 4, 5, 5, :], 'g-', linewidth=2, label='Pressure')
    plt.legend()
    plt.subplot(133)
    plt.plot(zc, Um[index, 3, 5, 5, :], 'r-', linewidth=2, label='Velocity')
    plt.legend()
    plt.show()
    """
