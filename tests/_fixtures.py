import numpy as np

from galerkin_neural_semigroup import NeuralSemigroupProblem
from ngfield import SimplicialDomain, Space, ZeroTrace, grad, inner


def heat_problem(*, dimension=2, radius=1.0, time_scale=1.0):
    vertices = np.linspace(0, 1, 9)[:, None]
    simplices = np.column_stack((np.arange(8), np.arange(1, 9)))
    geometry = SimplicialDomain(vertices, simplices)
    space = Space(
        geometry=geometry,
        components=1,
        restrictions=[ZeroTrace(component=0, boundary="all")],
    )
    basis = space.basis("laplacian", size=dimension, degree=1)

    def weak(u, v, dx, ds):
        return -0.01 * inner(grad(u[0]), grad(v[0])) * dx

    return NeuralSemigroupProblem(
        basis=basis,
        weak=weak,
        radius=radius,
        time_scale=time_scale,
    )
