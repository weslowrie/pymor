#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simplified version of the thermalblock demo with NGSolve as backend."""

from pymor.basic import *        # most common pyMOR functions and classes


PARAMS_PER_DOMAIN = 2
RBSIZE            = 20
TESTSIZE          = 20


def discretize():
    from ngsolve import (ngsglobals, Mesh, H1, CoefficientFunction, LinearForm, SymbolicLFI,
                         BilinearForm, SymbolicBFI, grad, TaskManager, GridFunction)
    from netgen.csg import CSGeometry, OrthoBrick, Pnt
    import numpy as np

    ngsglobals.msg_level = 1

    geo = CSGeometry()
    obox = OrthoBrick(Pnt(-1, -1, -1), Pnt(1, 1, 1)).bc("outer")

    b = []
    b.append(OrthoBrick(Pnt(-1, -1, -1), Pnt(0.0, 0.0, 0.0)).mat("mat1").bc("inner"))
    b.append(OrthoBrick(Pnt(-1,  0, -1), Pnt(0.0, 1.0, 0.0)).mat("mat2").bc("inner"))
    b.append(OrthoBrick(Pnt(0,  -1, -1), Pnt(1.0, 0.0, 0.0)).mat("mat3").bc("inner"))
    b.append(OrthoBrick(Pnt(0,   0, -1), Pnt(1.0, 1.0, 0.0)).mat("mat4").bc("inner"))
    b.append(OrthoBrick(Pnt(-1, -1,  0), Pnt(0.0, 0.0, 1.0)).mat("mat5").bc("inner"))
    b.append(OrthoBrick(Pnt(-1,  0,  0), Pnt(0.0, 1.0, 1.0)).mat("mat6").bc("inner"))
    b.append(OrthoBrick(Pnt(0,  -1,  0), Pnt(1.0, 0.0, 1.0)).mat("mat7").bc("inner"))
    b.append(OrthoBrick(Pnt(0,   0,  0), Pnt(1.0, 1.0, 1.0)).mat("mat8").bc("inner"))
    box = (obox - b[0] - b[1] - b[2] - b[3] - b[4] - b[5] - b[6] - b[7])

    geo.Add(box)
    for bi in b:
        geo.Add(bi)
    # domain 0 is empty!

    mesh = Mesh(geo.GenerateMesh(maxh=0.3))

    # H1-conforming finite element space
    V = H1(mesh, order=4, dirichlet="outer")
    v = V.TestFunction()
    u = V.TrialFunction()

    # Coeff as array: variable coefficient function (one CoefFct. per domain):
    sourcefct = CoefficientFunction([1 for i in range(9)])

    with TaskManager():
        # the right hand side
        f = LinearForm(V)
        f += SymbolicLFI(sourcefct * v)
        f.Assemble()
        gf_f = GridFunction(V)
        gf_f.vec.data = f.vec

        # the bilinear-form
        mats = []
        coeffs = [[0, 1, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 1, 0, 0, 0]]
        for c in coeffs:
            diffusion = CoefficientFunction(c)
            a = BilinearForm(V, symmetric=False)
            a += SymbolicBFI(diffusion * grad(u) * grad(v), definedon=(np.where(np.array(c) == 1)[0] + 1).tolist())
            a.Assemble()
            mats.append(a.mat)

    from pymor.bindings.ngsolve import NGSolveVectorSpace, NGSolveMatrixOperator, NGSolveVisualizer

    space = NGSolveVectorSpace(V)
    op = LincombOperator([NGSolveMatrixOperator(m, space, space) for m in mats],
                         [ProjectionParameterFunctional('diffusion', (len(coeffs),), (i,)) for i in range(len(coeffs))])

    h1_0_op = op.assemble([1] * len(coeffs)).with_(name='h1_0_semi')

    F = VectorFunctional(op.range.make_array([gf_f]))

    return StationaryDiscretization(op, F, visualizer=NGSolveVisualizer(mesh, V),
                                    products={'h1_0_semi': h1_0_op},
                                    parameter_space=CubicParameterSpace(op.parameter_type, 0.1, 1.))


if __name__ == '__main__':
    # build full order model
    ########################
    d = discretize()

    # select reduction algorithm with error estimator
    #################################################
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', d.parameter_type)
    reductor = CoerciveRBReductor(
        d, product=d.h1_0_semi_product,
        coercivity_estimator=coercivity_estimator
    )

    # generate reduced model
    ########################
    training_set = d.parameter_space.sample_uniformly(PARAMS_PER_DOMAIN)
    greedy_data = greedy(
        d, reductor, training_set,
        extension_params={'method': 'gram_schmidt'},
        max_extensions=RBSIZE
    )
    rd = greedy_data['reduced_discretization']

    # evaluate the reduction error
    ##############################
    results = reduction_error_analysis(rd, discretization=d, reductor=reductor, estimator=True,
                                       error_norms=[d.h1_0_semi_norm], condition=True,
                                       test_mus=TESTSIZE, random_seed=999, plot=True)

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
    U_RB = reductor.reconstruct(rd.solve(mumax))
    d.visualize((U, U_RB, U - U_RB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                separate_colorbars=True, block=True)
