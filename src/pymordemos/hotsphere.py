#!/usr/bin/env python
# encoding: utf-8

"""
Test problem demonstrating a 3D "Hot Sphere" in a stratified atmosphere domain.

This script evolves the 3D Euler equations.
The primary variables are:
    density (rho), x,y, and z momentum (rho*u,rho*v,rho*w), and energy.
"""
from __future__ import absolute_import
from clawpack import riemann
from pymor.bindings.pyclaw import *

import numpy as np
from mappedGrid import euler3d_mappedgrid as mg

# Test for MPI, and set sizes accordingly
try:
    from mpi4py import MPI
    mpiAvailable = True
except ImportError:
    raise ImportError('mpi4py is not available')
    mpiAvailable = False

if mpiAvailable:
    mpiRank = MPI.COMM_WORLD.Get_rank()
    mpiSize = MPI.COMM_WORLD.Get_size()
else:
    mpiRank = 0
    mpiSize = 1

# Constants
gamma = 1.4  # Ratio of Specific Heats
gamma1 = gamma - 1.
accelerationDueToGravity = 980.665  # Acceleration due to gravity in radial direction [cm/s**2]
kBoltzmann = 1.3807e-16  # Boltzmann constant [erg/K]
nAvogadro = 6.0221e23  # Avogadro's number [1/mol]
radiusOfEarth = 637120000.0  # Radius of Earth [cm]

# Hot Sphere Parameters
radiusOfHotSphere = 100.e5  # Radius of Hot Sphere
altitudeOfHotSphere = radiusOfEarth + 200.e5  # Altitude of Hot Sphere
mainSolver = 'classic'
filePrefix = 'spherical'
outDir = 'hotsphere_output'
neqn = 5
nwaves = 3
varNames = ['rho', 'rhou', 'rhov', 'rhow', 'energy']
auxNames = ['nxi', 'nyi', 'nzi', 'gi', 'nxj', 'nyj', 'nzj', 'gj', 'nxk', 'nyk', 'nzk',
            'gk', 'kappa', 'dr', 'r', 'gamman', 'g_r']
gravityTerm = True
roeEntropyFix = True  # <True/False> Roe entropy fix On or Off

# Time
output_style = 1
timeFinal = 20.0
numberOfOutputTimes = 1
out_times = np.array([0., 1., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120.])  # output_style=2

# Solver Options
theta0 = np.pi/4.0  # Center of Hot Sphere (theta angle)
phi0 = 0.15  # Center of Hot Sphere (phi angle)
xCenter = altitudeOfHotSphere*np.sin(theta0)*np.cos(phi0)  # x-coord for center of Hot Sphere
yCenter = altitudeOfHotSphere*np.sin(theta0)*np.sin(phi0)  # y-coord for center of Hot Sphere
zCenter = altitudeOfHotSphere*np.cos(theta0)               # z-coord for center of Hot Sphere

# Grid/Domain Parameters
mxyz = [160, 160, 80]  # Grid Resolution
xyzMin = [theta0-0.15, phi0-0.15, radiusOfEarth + 80.00e5]  # Domain lower limits
xyzMax = [theta0+0.15, phi0+0.15, radiusOfEarth + 950.0e5]  # Domain upper limits
mapType = "Spherical"


# -----------------------------------------------------------------------
# Description:
#   Equilibrium atmosphere
#
# Inputs:
#   p0[mz+2*mbc]      : pressure (1D array)
#   rho0[mz+2*mbc]    : density (1D array)
#   Mavg[mz+2*mbc]    : average molecular mass (1D array)
#
# Input/Outputs:
#   p0,rho0,Mavg      : 1D z-column initialization of p0 and rho0
# -----------------------------------------------------------------------
def setEquilibriumAtmosphere(p0, rho0, Mavg):

    p0 = [1.28255457e+02,2.45768842e+01,4.14947876e+00,6.29750420e-01,1.01220380e-01,2.64133921e-02,1.22941741e-02,7.08667395e-03,4.52931611e-03,3.07286214e-03,2.16905463e-03,1.57652477e-03,1.17092484e-03,8.84611067e-04,6.77691403e-04,5.25138237e-04,4.10841768e-04,3.24102394e-04,2.57470120e-04,2.05925021e-04,1.65598592e-04,1.33701518e-04,1.08364754e-04,8.82441931e-05,7.21143717e-05,5.91376054e-05,4.86178229e-05,4.00787900e-05,3.30908693e-05,2.73888126e-05,2.27031016e-05,1.88518481e-05,1.56898948e-05,1.30700401e-05,1.08991559e-05,9.09869161e-06,7.60521743e-06,6.36376491e-06,5.32972657e-06,4.46856235e-06,3.74878325e-06,3.14890785e-06,2.64613146e-06,2.22646032e-06,1.87396531e-06,1.57844875e-06,1.33028392e-06,1.12211091e-06,9.47071388e-07,7.99762122e-07,6.75921511e-07,5.71493939e-07,4.83610358e-07,4.09325094e-07,3.46744110e-07,2.93793938e-07,2.49152408e-07,2.11367113e-07,1.79432411e-07,1.52415843e-07,1.29549499e-07,1.10136422e-07,9.37086690e-08,7.97324669e-08,6.79127210e-08,5.78532722e-08,4.93172661e-08,4.20604343e-08,3.58836884e-08,3.06389102e-08,2.61608771e-08,2.23557534e-08,1.91042726e-08,1.63479490e-08,1.39976779e-08,1.19853352e-08,1.02623231e-08,8.78713846e-09,7.53940212e-09,6.46885245e-09,5.55032464e-09,4.76222864e-09,4.09020086e-09,3.51658796e-09]

    rho0 = [1.93347036e-07,4.03984315e-08,7.33795328e-09,1.16964004e-09,1.64049100e-10,2.53990286e-11,7.54287116e-12,3.40478277e-12,1.84556481e-12,1.10964372e-12,7.13581470e-13,4.81506393e-13,3.36472592e-13,2.41540079e-13,1.77156053e-13,1.32213794e-13,1.00089557e-13,7.67024111e-14,5.93930647e-14,4.64294817e-14,3.65782332e-14,2.90138753e-14,2.31378048e-14,1.85800114e-14,1.49929512e-14,1.21526733e-14,9.89015561e-15,8.07840567e-15,6.61976992e-15,5.43890503e-15,4.48202167e-15,3.70250573e-15,3.06590093e-15,2.54266886e-15,2.11283102e-15,1.75827860e-15,1.46560471e-15,1.22337830e-15,1.02239821e-15,8.55585508e-16,7.16578299e-16,6.01033981e-16,5.04419184e-16,4.23940996e-16,3.56468062e-16,2.99992883e-16,2.52633808e-16,2.12955966e-16,1.79630105e-16,1.51610996e-16,1.28075790e-16,1.08244792e-16,9.15665290e-17,7.74771188e-17,6.56137471e-17,5.55805979e-17,4.71251502e-17,3.99708405e-17,3.39261636e-17,2.88137888e-17,2.44878021e-17,2.08159094e-17,1.77092661e-17,1.50666724e-17,1.28321441e-17,1.09306468e-17,9.31730480e-18,7.94587120e-18,6.77866202e-18,5.78764327e-18,4.94156316e-18,4.22266806e-18,3.60840539e-18,3.08771188e-18,2.64374425e-18,2.26362608e-18,1.93817162e-18,1.65953699e-18,1.42386938e-18,1.22167290e-18,1.04819271e-18,8.99349679e-19,7.72429901e-19,6.64098458e-19]

    Mavg = [28.85614554,28.85337155,28.83817654,28.56226512,27.60224909,26.26692289,25.23573593,24.45469565,23.79308533,23.18781005,22.61490394,22.07318988,21.55703223,21.06778441,20.60540309,20.17202267,19.76585711,19.38847601,19.0408475, 18.71970337,18.42758099,18.16274099,17.92359740,17.70606183,17.51035814,17.33530373,17.17893585,17.03979933,16.91620578,16.80712079,16.71028376,16.62471452,16.54940299,16.48292773,16.42454596,16.37307369,16.32776306,16.28801338,16.2531155, 16.22247335,16.19551611,16.17188138,16.15108306,16.13288090,16.11686426,16.10282002,16.09046507,16.07960946,16.07007411,16.06169374,16.05433222,16.04784993,16.04215209,16.03712679,16.0327204,16.02883120,16.02540929,16.02239140,16.01973516,16.01738918,16.01531699,16.01348647,16.01187781,16.01045286,16.00919766,16.00808580,16.00710454,16.00623687,16.00546792,16.00478755,16.00418349,16.00365220,16.00317996,16.00276269,16.00239247,16.00206303,16.00176987,16.00150902,16.00127962,16.00107519,16.00089299,16.00073063,16.00058692,16.00045964]

    return p0, rho0, Mavg


# -----------------------------------------------------------------------
# Description:
#   Modify pressure to create numeric atmosphere equilibrium
#
# Inputs:
#   ze0[mz+2*mbc+1]     : cell edge grid values
#   p0[mz+2*mbc]        : pressure
#   rho0[mz+2*mbc]      : density
#
# Input/Outputs:
#   p0,rho0           : 1D z-column modification of p0 and rho0
# -----------------------------------------------------------------------
def modifyEquilibriumAtmosphere(zep0, p0, rho0):

    # Compute the delta-z (dz)
    nz = np.size(zep0)-1
    dz = np.zeros([nz], dtype='float', order='F')
    for iz in range(nz-1):
        dz[iz] = zep0[iz+1]-zep0[iz]

    # Compute modified pressure at cell centers
    iz = nz-1
    dz2 = (dz[iz]+dz[iz-1])*0.5
    p0[iz] = p0[iz] + rho0[iz]*accelerationDueToGravity*dz2
    for iz in range(nz-1, 0, -1):
        dz2 = (dz[iz]+dz[iz-1])*0.5
        finterp = dz[iz-1]/(dz[iz]+dz[iz-1])
        rho_b = rho0[iz]*finterp + rho0[iz-1]*(1.-finterp)
        p0[iz-1] = p0[iz] + rho_b*accelerationDueToGravity*dz2

    return p0


# -----------------------------------------------------------------------
# Description:
#   Custom BCs for the z-direction
# -----------------------------------------------------------------------
def customBCLowerZ(state, dim, t, qbc, auxbc, mbc):
    for k in range(mbc):
        rZ = np.sqrt(xcpZ[k]**2 + ycpZ[k]**2 + zcpZ[k]**2)-radiusOfEarth
        qbc[0, :, :, k] = rho0[k]
        qbc[1, :, :, k] = qbc[1, :, :, mbc]
        qbc[2, :, :, k] = qbc[2, :, :, mbc]
        qbc[3, :, :, k] = qbc[3, :, :, mbc]
        rhov2 = (qbc[1, :, :, k]**2+qbc[2, :, :, k]**2+qbc[3, :, :, k]**2)/qbc[0, :, :, k]
        qbc[4, :, :, k] = p0[k]/gamma1 + 0.5*rhov2


def customBCUpperZ(state, dim, t, qbc, auxbc, mbc):
    for k in range(mbc):
        # rZ = np.sqrt(xcpZ[-k-1]**2 + ycpZ[-k-1]**2 + zcpZ[-k-1]**2)-radiusOfEarth
        # qbc[0,:,:,-k-1] = rho0[-k-1]
        # qbc[1,:,:,-k-1] = qbc[1,:,:,-mbc-1]
        # qbc[2,:,:,-k-1] = qbc[2,:,:,-mbc-1]
        # qbc[3,:,:,-k-1] = qbc[3,:,:,-mbc-1]
        # rhov2 = (qbc[1,:,:,-k-1]**2 + qbc[2,:,:,-k-1]**2 + qbc[3,:,:,-k-1]**2)/qbc[0,:,:,-k-1]
        # qbc[4,:,:,-k-1] = p0[-k-1]/gamma1 + 0.5*rhov2

        # Fractional BC with delta=0 if negative
        deltar = qbc[0, :, :, -mbc-1] - rho0[-mbc-1]
        fracr = deltar/rho0[-mbc-1]
        fracr = np.nan_to_num(fracr)
        fracrgt0 = np.greater(fracr, 0.0)
        qbc[0, :, :, -k-1] = rho0[-k-1] + deltar*fracrgt0
        qbc[1, :, :, -k-1] = qbc[0, :, :, -k-1]*qbc[1, :, :, -mbc-1]/qbc[0, :, :, -mbc-1]
        qbc[2, :, :, -k-1] = qbc[0, :, :, -k-1]*qbc[2, :, :, -mbc-1]/qbc[0, :, :, -mbc-1]
        qbc[3, :, :, -k-1] = qbc[0, :, :, -k-1]*qbc[3, :, :, -mbc-1]/qbc[0, :, :, -mbc-1]
        rhov2 = (qbc[1, :, :, -mbc-1]**2+qbc[2, :, :, -mbc-1]**2+qbc[3, :, :, -mbc-1]**2)/qbc[0, :, :, -mbc-1]
        deltap = gamma1*(qbc[4, :, :, -mbc-1]-0.5*rhov2)-p0[-mbc-1]
        fracp = deltap/p0[-mbc-1]
        fracp = np.nan_to_num(fracp)
        fracpgt0 = np.greater(fracp, 0.0)
        rhov2 = (qbc[1, :, :, -k-1]**2+qbc[2, :, :, -k-1]**2+qbc[3, :, :, -k-1]**2)/qbc[0, :, :, -k-1]
        qbc[4, :, :, -k-1] = (p0[-k-1] + deltap*fracpgt0)/gamma1 + 0.5*rhov2


def customAuxBCLowerZ(state, dim, t, qbc, auxbc, mbc):
    auxbc[:, :, :, :mbc] = auxtmp[:, :, :, :mbc]


def customAuxBCUpperZ(state, dim, t, qbc, auxbc, mbc):
    auxbc[:, :, :, -mbc:] = auxtmp[:, :, :, -mbc:]


# -------------------------------------------------------------------------------
# Main Script for "Hot Sphere" Problem
# -------------------------------------------------------------------------------
def hotsphere3D(kernel_language='Fortran', solver_type=mainSolver,
                use_petsc=False, outdir=outDir,
                output_format='hdf5', file_prefix=filePrefix,
                disable_output=False, mx=mxyz[0], my=mxyz[1], mz=mxyz[2],
                tfinal=timeFinal, num_output_times=numberOfOutputTimes):

    # Load PyClaw
    if mpiSize > 1 and not use_petsc:
        print("For MPI runs, use_petsc=True, exiting.")
        exit()

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    # Solver Settings
    if solver_type == 'classic':
        import euler_3d_gmap
        solver = pyclaw.ClawSolver3D()
        solver.rp = euler_3d_gmap
        solver.num_eqn = neqn
        solver.num_waves = nwaves
        solver.dimensional_split = True
        solver.limiters = pyclaw.limiters.tvd.minmod
        solver.num_ghost = 2
        solver.order = 2
        solver.fwave = True
        solver.source_split = 0
        solver.before_step = None
        solver.step_source = None
        solver.dt_variable = True
        solver.cfl_max = 0.60
        solver.cfl_desired = 0.50
        solver.dt_initial = 1.e-4
        solver.max_steps = 50000
    else:
        raise Exception('Unrecognized solver_type')

    # Logging
    import logging
    solver.logger.setLevel(logging.DEBUG)
    solver.logger.info("PyClaw Solver: "+solver_type)

    # Domain
    x = pyclaw.Dimension(0.0, 1.0, mx, name='x')
    y = pyclaw.Dimension(0.0, 1.0, my, name='y')
    z = pyclaw.Dimension(0.0, 1.0, mz, name='z')
    domain = pyclaw.Domain([x, y, z])
    num_aux = 15
    state = pyclaw.State(domain, solver.num_eqn, num_aux)

    # Define variables passed to the Riemann solver via Fortran COMMON block
    state.problem_data['gamma'] = gamma
    state.problem_data['g_r'] = accelerationDueToGravity
    state.problem_data['gravity'] = gravityTerm
    state.problem_data['gravityflux'] = False

    # Grids
    mbc = solver.num_ghost
    grid = state.grid

    dxc, dyc, dzc = domain.grid.delta[0], domain.grid.delta[1], domain.grid.delta[2]
    pmx, pmy, pmz = grid.num_cells[0], grid.num_cells[1], grid.num_cells[2]
    pmxBC, pmyBC, pmzBC = pmx+2*mbc, pmy+2*mbc, pmz+2*mbc

    centers = grid.c_centers  # cell centers
    centersBC = grid.c_centers_with_ghost(mbc)
    nodesBC = grid.c_nodes_with_ghost(mbc)

    Xcc, Ycc, Zcc = centers[0][:][:][:], centers[1][:][:][:], centers[2][:][:][:]
    Xcp, Ycp, Zcp = mg.mapc2pwrapper(Xcc, Ycc, Zcc, pmz, xyzMin, xyzMax, mapType)
    Xcp = np.reshape(Xcp, [pmx, pmy, pmz], order='F')  # x centers (Phys.)
    Ycp = np.reshape(Ycp, [pmx, pmy, pmz], order='F')  # y centers (Phys.)
    Zcp = np.reshape(Zcp, [pmx, pmy, pmz], order='F')  # z centers (Phys.)

    # Grid nodes With Boundary Cells (1D Slice along z)- Comp. and Phys.
    xecZ = nodesBC[0][0][0][:]  # x nodes along z (Comp.)
    yecZ = nodesBC[1][0][0][:]  # y nodes along z (Comp.)
    zecZ = nodesBC[2][0][0][:]  # z nodes along z (Comp.)
    xepZ, yepZ, zepZ = mg.mapc2pwrapper(xecZ, yecZ, zecZ, pmz, xyzMin, xyzMax, mapType)

    # Grid Centers With Boundary Cells (1D Slice along z) - Comp. and Phys.
    global xcpZ, ycpZ, zcpZ
    xccZ = centersBC[0][0][0][:]  # x centers along z (Comp.)
    yccZ = centersBC[1][0][0][:]  # y centers along z (Comp.)
    zccZ = centersBC[2][0][0][:]  # z centers along z (Comp.)
    xcpZ, ycpZ, zcpZ = mg.mapc2pwrapper(xccZ, yccZ, zccZ, pmz, xyzMin, xyzMax, mapType)

    # Equilibrium Atmosphere
    global p0, rho0, Mavg
    p0 = np.zeros([pmzBC], dtype='float', order='F')
    rho0 = np.zeros([pmzBC], dtype='float', order='F')
    Mavg = np.zeros([pmzBC], dtype='float', order='F')
    p0, rho0, Mavg = setEquilibriumAtmosphere(p0, rho0, Mavg)  # Set the equilibrium pressure such that dp/dz = -rho*gR
    altEdgesAboveEarth = np.sqrt(xepZ ** 2 + yepZ ** 2 + zepZ ** 2) - radiusOfEarth  # Modify the equilibrium such that dp/dz = -rho*gR is held numerically
    p0 = modifyEquilibriumAtmosphere(altEdgesAboveEarth, p0, rho0)

    # Aux Variables
    xlower, ylower, zlower = nodesBC[0][0][0][0], nodesBC[1][0][0][0], nodesBC[2][0][0][0]
    global auxtmp
    auxtmp = np.zeros([num_aux, pmx + 2 * mbc, pmy + 2 * mbc, pmz + 2 * mbc], dtype='float', order='F')
    auxtmp = mg.setauxiliaryvariables(num_aux, mbc, pmx, pmy, pmz, xlower, ylower, zlower, dxc, dyc, dzc, xyzMin,
                                      xyzMax, mapType)
    state.aux[:, :, :, :] = auxtmp[:, mbc:-mbc, mbc:-mbc, mbc:-mbc]

    # State Variables
    q = state.q
    p = np.zeros([pmx, pmy, pmz], dtype='float', order='F')
    T = np.zeros([pmx, pmy, pmz], dtype='float', order='F')
    for i in range(np.size(p, 0)):
        for j in range(np.size(p, 1)):
            p[i, j, :] = p0[mbc:-mbc]
            q[0, i, j, :] = rho0[mbc:-mbc]
    q[1, :, :, :] = 0.
    q[2, :, :, :] = 0.
    q[3, :, :, :] = 0.
    q[4, :, :, :] = p/gamma1

    # Add Perturbation
    T[:, :, :] = p / state.q[0, :, :, :]
    L = np.sqrt((Xcp-xCenter)**2 + (Ycp-yCenter)**2 + (Zcp-zCenter)**2)
    for i in range(pmx):
        for j in range(pmy):
            for k in range(pmz):
                if L[i, j, k] <= radiusOfHotSphere:
                    # mu = Mavg[k + mbc] / nAvogadro
                    # T[i,j,k] += 11.604e3*(kBoltzmann/mu)*(1.0-L[i,j,k]/radiusOfHotSphere)
                    # p[i,j,k] = T[i,j,k]*state.q[0,i,j,k]
                    # print("i:"+str(i)+" j:"+str(j)+" k:"+str(k))
                    p[i, j, k] += 11.604e3*kBoltzmann*nAvogadro/Mavg[k+mbc]*q[0, i, j, k]\
                        * (1.0-L[i, j, k]/radiusOfHotSphere)
    q[4, :, :, :] = p/gamma1

    # solver.logger.info("Temperature Min/Max: "+str(np.min(p*Mavg/nAvogadro/kBoltzmann))+"/"+str(np.max(p*Mavg/nAvogadro/kBoltzmann)) )
    # solver.logger.info("Temperature Min/Max: " + str(np.min(T)) + "/" + str(np.max(T)))
    
    # Index for Capacity function in state.aux (Python 0-based)
    state.index_capa = 12

    # Boundary Conditions
    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.bc_lower[1] = pyclaw.BC.extrap
    solver.bc_upper[1] = pyclaw.BC.extrap
    solver.bc_lower[2] = pyclaw.BC.custom
    solver.bc_upper[2] = pyclaw.BC.custom
    solver.user_bc_lower = customBCLowerZ
    solver.user_bc_upper = customBCUpperZ

    # Aux - Boundary Conditions
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[1] = pyclaw.BC.extrap
    solver.aux_bc_upper[1] = pyclaw.BC.extrap
    solver.aux_bc_lower[2] = pyclaw.BC.custom
    solver.aux_bc_upper[2] = pyclaw.BC.custom
    solver.user_aux_bc_lower = customAuxBCLowerZ
    solver.user_aux_bc_upper = customAuxBCUpperZ

    # Setup Controller
    claw = pyclaw.Controller()
    claw.verbosity = 4
    claw.solution = pyclaw.Solution(state, domain)
    claw.write_aux_init = True
    claw.solver = solver
    claw.output_format = output_format
    claw.output_file_prefix = file_prefix
    if not use_petsc: claw.output_options = {'compression': 'gzip'}
    claw.write_aux_always = False
    claw.keep_copy = True
    claw.tfinal = tfinal
    claw.num_output_times = num_output_times
    claw.outdir = outdir
    claw.output_style = output_style
    if output_style == 2:
        claw.out_times = out_times

    # Print output times
    if output_style == 1:
        outTimes = np.linspace(claw.solution.t, tfinal, num_output_times+1)
    elif output_style == 2:
        outTimes = claw.out_times
    print("Planned output times: ["+" ".join(str(e) for e in outTimes)+"]")

    return PyClawModel(claw)


def writeROMtoFile(U,modes,outDir):
    import h5py
    h5FileName = outDir + '/' + 'ROM.h5'
    f = h5py.File(h5FileName, 'w')
    Un = U.to_numpy()
    dset = f.create_dataset('U', (np.size(Un, 0), np.size(Un, 1), ), 'f8', compression='gzip')
    dset[...] = Un
    modesn = modes.to_numpy()
    dset = f.create_dataset('modes', (np.size(modesn, 0), np.size(modesn, 1), ), 'f8', compression='gzip')
    dset[...] = modesn
    f.close()


# -------------------------------------------------------------------------------
# __main__()
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    from pymor.algorithms.pod import pod
    from pymor.bindings.pyclaw import *
    from pymor.algorithms.timestepping import ExplicitEulerTimeStepper
    import matplotlib.pyplot as plt
    import math
    problem = hotsphere3D(kernel_language='Fortran', solver_type=mainSolver,
                          use_petsc=False, outdir=outDir, output_format='hdf5', file_prefix=filePrefix,
                          disable_output=False, mx=mxyz[0], my=mxyz[1], mz=mxyz[2],
                          tfinal=timeFinal, num_output_times=numberOfOutputTimes)

    q0 = np.reshape(problem.claw.solution.state.q, problem.claw.solution.state.q.size, order='F')
    q0_array = NumpyVectorSpace(problem.claw.solution.state.q.size).make_array([q0])
    source_space = NumpyVectorSpace(problem.claw.solution.state.q.size)
    source_space.make_array([q0])
    range_space = NumpyVectorSpace(problem.claw.solution.state.q.size)
    op = PyClawOperator(problem.claw, source_space, range_space)
    vis = PyClawVisualizer(problem.claw)
    ts = ExplicitEulerTimeStepper(nt=480)
    problem.claw.solver.setup(problem.claw.solution)

    fom = InstationaryModel(T=240., initial_data=q0_array, operator=op,
                            rhs=None, mass=None, visualizer=vis, time_stepper=ts,
                            num_values=80)

    # Constant Timesteps
    dt = fom.T/fom.time_stepper.nt
    problem.claw.solver.dt = dt

    U = fom.solve(mu=None)  # Solve
    modes, svals = pod(U)
    print(np.shape(modes.to_numpy()))
    print(svals)

    ## Write ROM to File
    # writeROMtoFile(U, modes, outDir)

    # Plot the singular values
    plt.figure()
    plt.semilogy(svals, linewidth=2, label='Singular Values')
    plt.legend()
    plt.show()

    fom.visualize(U)  # plot the solution

    num_output_times = fom.time_stepper.nt
    print("number of output:", num_output_times)

    from pymor.algorithms.basic import *
    # Up = modes[0:2].lincomb(U.inner(modes[0:2],product=None))
    Up = project_array(U, modes[0:2])
    fom.visualize(Up)

    Up = project_array(U, modes[0:5])
    fom.visualize(Up)

    Up = project_array(U, modes[0:8])
    fom.visualize(Up)

    Up = project_array(U, modes[0:11])
    fom.visualize(Up)

    Up = project_array(U, modes[0:num_output_times+1])
    fom.visualize(Up)
