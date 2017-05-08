#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simplified version of the thermalblock demo.

Usage:
  thermalblock_simple.py [options] MODEL ALG SNAPSHOTS RBSIZE TEST

Arguments:
  MODEL      High-dimensional model (pymor, dunegdt, fenics).

  ALG        The model reduction algorithm to use
             (naive, greedy, adaptive_greedy, pod).

  SNAPSHOTS  naive:           ignored

             greedy/pod:      Number of training_set parameters per block
                              (in total SNAPSHOTS^(XBLOCKS * YBLOCKS)
                              parameters).

             adaptive_greedy: size of validation set.

  RBSIZE     Size of the reduced basis.

  TEST       Number of parameters for stochastic error estimation.
"""

from pymor.basic import *        # most common pyMOR functions and classes
from functools import partial    # fix parameters of given function


# parameters for high-dimensional models
XBLOCKS = 2
YBLOCKS = 2
GRID_INTERVALS = 100
FENICS_ORDER = 2


####################################################################################################
# High-dimensional models                                                                          #
####################################################################################################


def discretize_pymor():

    # setup analytical problem
    problem = thermal_block_problem(num_blocks=(XBLOCKS, YBLOCKS))

    # discretize using continuous finite elements
    d, _ = discretize_stationary_cg(problem, diameter=1. / GRID_INTERVALS)

    return d


def discretize_fenics():
    from pymor.tools import mpi

    if mpi.parallel:
        from pymor.discretizations.mpi import mpi_wrap_discretization
        return mpi_wrap_discretization(_discretize_fenics, use_with=True, pickle_local_spaces=False)
    else:
        return _discretize_fenics()


def _discretize_fenics():

    # assemble system matrices - FEniCS code
    ########################################

    import dolfin as df

    mesh = df.UnitSquareMesh(GRID_INTERVALS, GRID_INTERVALS, 'crossed')
    V = df.FunctionSpace(mesh, 'Lagrange', FENICS_ORDER)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    diffusion = df.Expression('(lower0 <= x[0]) * (open0 ? (x[0] < upper0) : (x[0] <= upper0)) *' +
                              '(lower1 <= x[1]) * (open1 ? (x[1] < upper1) : (x[1] <= upper1))',
                              lower0=0., upper0=0., open0=0,
                              lower1=0., upper1=0., open1=0,
                              element=df.FunctionSpace(mesh, 'DG', 0).ufl_element())

    def assemble_matrix(x, y, nx, ny):
        diffusion.user_parameters['lower0'] = x/nx
        diffusion.user_parameters['lower1'] = y/ny
        diffusion.user_parameters['upper0'] = (x + 1)/nx
        diffusion.user_parameters['upper1'] = (y + 1)/ny
        diffusion.user_parameters['open0'] = (x + 1 == nx)
        diffusion.user_parameters['open1'] = (y + 1 == ny)
        return df.assemble(df.inner(diffusion * df.nabla_grad(u), df.nabla_grad(v)) * df.dx)

    mats = [assemble_matrix(x, y, XBLOCKS, YBLOCKS)
            for x in range(XBLOCKS) for y in range(YBLOCKS)]
    mat0 = mats[0].copy()
    mat0.zero()
    h1_mat = df.assemble(df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx)

    f = df.Constant(1.) * v * df.dx
    F = df.assemble(f)

    bc = df.DirichletBC(V, 0., df.DomainBoundary())
    for m in mats:
        bc.zero(m)
    bc.apply(mat0)
    bc.apply(h1_mat)
    bc.apply(F)

    # wrap everything as a pyMOR discretization
    ###########################################

    # FEniCS wrappers
    from pymor.bindings.fenics import FenicsVectorSpace, FenicsMatrixOperator, FenicsVisualizer

    # define parameter functionals (same as in pymor.analyticalproblems.thermalblock)
    parameter_functionals = [ProjectionParameterFunctional(component_name='diffusion',
                                                           component_shape=(YBLOCKS, XBLOCKS),
                                                           coordinates=(YBLOCKS - y - 1, x))
                             for x in range(XBLOCKS) for y in range(YBLOCKS)]

    # wrap operators
    ops = [FenicsMatrixOperator(mat0, V, V)] + [FenicsMatrixOperator(m, V, V) for m in mats]
    op = LincombOperator(ops, [1.] + parameter_functionals)
    rhs = VectorFunctional(FenicsVectorSpace(V).make_array([F]))
    h1_product = FenicsMatrixOperator(h1_mat, V, V, name='h1_0_semi')

    # build discretization
    visualizer = FenicsVisualizer(FenicsVectorSpace(V))
    parameter_space = CubicParameterSpace(op.parameter_type, 0.1, 1.)
    d = StationaryDiscretization(op, rhs, products={'h1_0_semi': h1_product},
                                 parameter_space=parameter_space,
                                 visualizer=visualizer)

    return d


def discretize_dunegdt():

    from itertools import product
    from pymor.core.config import config
    import numpy as np

    assert config.HAVE_DUNEXT
    assert config.HAVE_DUNEGDT

    # assemble system matrices - dune-gdt code
    ##########################################

    from dune.xt.common import init_mpi
    init_mpi()

    from dune.xt.grid import HAVE_DUNE_ALUGRID
    assert HAVE_DUNE_ALUGRID
    from dune.xt.grid import make_cube_grid__2d_simplex_aluconform, make_boundary_info
    grid = make_cube_grid__2d_simplex_aluconform(lower_left=[0, 0], upper_right=[1, 1],
                                                 num_elements=[GRID_INTERVALS, GRID_INTERVALS],
                                                 num_refinements=1, overlap_size=[0, 0])
    boundary_info = make_boundary_info(grid, 'xt.grid.boundaryinfo.alldirichlet')

    from dune.xt.functions import make_checkerboard_function_1x1, make_constant_function_1x1

    def diffusion_function_factory(ix, iy):
        values = [[0.]]*(YBLOCKS*XBLOCKS)
        values[ix + XBLOCKS*iy] = [1.]
        return make_checkerboard_function_1x1(grid_provider=grid, lower_left=[0, 0], upper_right=[1, 1],
                                              num_elements=[XBLOCKS, YBLOCKS],
                                              values=values, name='diffusion_{}_{}'.format(ix, iy))

    diffusion_functions = [diffusion_function_factory(ix, iy)
                           for ix, iy in product(range(XBLOCKS), range(YBLOCKS))]
    one = make_constant_function_1x1(grid, 1.)

    from dune.gdt import HAVE_DUNE_FEM
    assert HAVE_DUNE_FEM
    from dune.gdt import (make_cg_leaf_to_1x1_fem_p1_space,
                          make_elliptic_matrix_operator_istl_row_major_sparse_matrix_double,
                          make_dirichlet_constraints,
                          make_l2_volume_vector_functional_istl_dense_vector_double,
                          make_system_assembler,
                          HAVE_DUNE_FEM)

    space = make_cg_leaf_to_1x1_fem_p1_space(grid)
    system_assembler = make_system_assembler(space)

    from dune.xt.la import HAVE_DUNE_ISTL
    assert HAVE_DUNE_ISTL
    from dune.xt.la import IstlDenseVectorDouble

    elliptic_operators = [make_elliptic_matrix_operator_istl_row_major_sparse_matrix_double(diffusion_function, space)
                          for diffusion_function in diffusion_functions]
    for op in elliptic_operators:
        system_assembler.append(op)

    l2_force_functional = make_l2_volume_vector_functional_istl_dense_vector_double(one, space)
    system_assembler.append(l2_force_functional)

    h1_product_operator = make_elliptic_matrix_operator_istl_row_major_sparse_matrix_double(one, space)
    system_assembler.append(h1_product_operator)

    clear_dirichlet_rows = make_dirichlet_constraints(boundary_info, space.size(), False)
    set_dirichlet_rows = make_dirichlet_constraints(boundary_info, space.size(), True)
    system_assembler.append(clear_dirichlet_rows)
    system_assembler.append(set_dirichlet_rows)

    system_assembler.assemble()

    rhs_vector = l2_force_functional.vector()
    lhs_matrices = [op.matrix() for op in elliptic_operators]
    for mat in lhs_matrices:
        clear_dirichlet_rows.apply(mat)
    lhs_matrices.append(lhs_matrices[0].copy())
    lhs_matrices[-1].scal(0)
    set_dirichlet_rows.apply(lhs_matrices[-1])
    h1_0_matrix = h1_product_operator.matrix()
    set_dirichlet_rows.apply(h1_0_matrix)
    set_dirichlet_rows.apply(rhs_vector)

    # wrap everything as a pyMOR discretization
    ###########################################

    # dune-xt-la wrappers
    from pymor.bindings.dunext import DuneXTVectorSpace, DuneXTMatrixOperator
    # dune-gdt wrappers
    from pymor.bindings.dunegdt import DuneGDTVisualizer, DuneGDTpyMORVisualizerWrapper

    # define parameter functionals (same as in pymor.analyticalproblems.thermalblock)
    parameter_functionals = [ProjectionParameterFunctional(component_name='diffusion',
                                                           component_shape=(YBLOCKS, XBLOCKS),
                                                           coordinates=(YBLOCKS - y - 1, x))
                             for x in range(XBLOCKS) for y in range(YBLOCKS)]

    # wrap operators
    ops = [DuneXTMatrixOperator(mat) for mat in lhs_matrices]
    op = LincombOperator(ops, parameter_functionals + [1.])
    rhs = VectorFunctional(DuneXTVectorSpace(IstlDenseVectorDouble, space.size()).make_array([rhs_vector]))
    h1_product = DuneXTMatrixOperator(h1_0_matrix, name='h1_0_semi')

    # build visualizer and discretization
    visualizer = DuneGDTVisualizer(space)
    parameter_space = CubicParameterSpace(op.parameter_type, 0.1, 1.)
    d = StationaryDiscretization(op, rhs, products={'h1_0_semi': h1_product},
                                 parameter_space=parameter_space,
                                 visualizer=visualizer)

    return d


####################################################################################################
# Reduction algorithms                                                                             #
####################################################################################################


def reduce_naive(d, reductor, basis_size):

    training_set = d.parameter_space.sample_randomly(basis_size)

    snapshots = d.operator.source.empty()
    for mu in training_set:
        snapshots.append(d.solve(mu))

    rd, rc, _ = reductor(d, snapshots)

    return rd, rc


def reduce_greedy(d, reductor, snapshots, basis_size):

    training_set = d.parameter_space.sample_uniformly(snapshots)
    extension_algorithm = partial(gram_schmidt_basis_extension, product=d.h1_0_semi_product)
    pool = new_parallel_pool()

    greedy_data = greedy(d, reductor, training_set,
                         extension_algorithm=extension_algorithm, max_extensions=basis_size,
                         pool=pool)

    return greedy_data['reduced_discretization'], greedy_data['reconstructor']


def reduce_adaptive_greedy(d, reductor, validation_mus, basis_size):

    extension_algorithm = partial(gram_schmidt_basis_extension, product=d.h1_0_semi_product)
    pool = new_parallel_pool()

    greedy_data = adaptive_greedy(d, reductor, validation_mus=-validation_mus,
                                  extension_algorithm=extension_algorithm, max_extensions=basis_size,
                                  pool=pool)

    return greedy_data['reduced_discretization'], greedy_data['reconstructor']


def reduce_pod(d, reductor, snapshots, basis_size):

    training_set = d.parameter_space.sample_uniformly(snapshots)

    snapshots = d.operator.source.empty()
    for mu in training_set:
        snapshots.append(d.solve(mu))

    basis, singular_values = pod(snapshots, modes=basis_size, product=d.h1_0_semi_product)

    rd, rc, _ = reductor(d, basis)

    return rd, rc


####################################################################################################
# Main script                                                                                      #
####################################################################################################

def main():
    # command line argument parsing
    ###############################
    import sys
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)
    MODEL, ALG, SNAPSHOTS, RBSIZE, TEST = sys.argv[1:]
    MODEL, ALG, SNAPSHOTS, RBSIZE, TEST = MODEL.lower(), ALG.lower(), int(SNAPSHOTS), int(RBSIZE), int(TEST)

    # discretize
    ############
    if MODEL == 'pymor':
        d = discretize_pymor()
    elif MODEL == 'fenics':
        d = discretize_fenics()
    elif MODEL == 'dunegdt':
        d = discretize_dunegdt()
    else:
        raise NotImplementedError

    # select reduction algorithm with error estimator
    #################################################
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', d.parameter_type)
    reductor = partial(reduce_coercive,
                       product=d.h1_0_semi_product, coercivity_estimator=coercivity_estimator)

    # generate reduced model
    ########################
    if ALG == 'naive':
        rd, rc = reduce_naive(d, reductor, RBSIZE)
    elif ALG == 'greedy':
        rd, rc = reduce_greedy(d, reductor, SNAPSHOTS, RBSIZE)
    elif ALG == 'adaptive_greedy':
        rd, rc = reduce_adaptive_greedy(d, reductor, SNAPSHOTS, RBSIZE)
    elif ALG == 'pod':
        rd, rc = reduce_pod(d, reductor, SNAPSHOTS, RBSIZE)
    else:
        raise NotImplementedError

    # evaluate the reduction error
    ##############################
    results = reduction_error_analysis(rd, discretization=d, reconstructor=rc, estimator=True,
                                       error_norms=[d.h1_0_semi_norm], condition=True,
                                       test_mus=TEST, random_seed=999, plot=True)

    # show results
    ##############
    print(results['summary'])
    import matplotlib.pyplot
    matplotlib.pyplot.show(results['figure'])

    # write results to disk
    #######################
    from pymor.core.pickle import dump
    dump(rd, open('reduced_model.out', 'wb'))
    results.pop('figure')  # matplotlib figures cannot be serialized
    dump(results, open('results.out', 'wb'))

    # visualize reduction error for worst-approximated mu
    #####################################################
    mumax = results['max_error_mus'][0, -1]
    U = d.solve(mumax)
    U_RB = rc.reconstruct(rd.solve(mumax))
    d.visualize((U, U_RB, U - U_RB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                separate_colorbars=True, block=True)


if __name__ == '__main__':
    main()
