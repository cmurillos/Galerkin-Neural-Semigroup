"""Minimal end-to-end heat example using the compact public API."""

import numpy as np
import torch
from ngfield import SimplicialDomain, Space, ZeroTrace, grad, inner

from galerkin_neural_semigroup import NeuralSemigroupProblem


def main():
    vertices = np.linspace(0, 1, 17)[:, None]
    simplices = np.column_stack((np.arange(16), np.arange(1, 17)))
    geometry = SimplicialDomain(vertices, simplices)
    space = Space(
        geometry=geometry,
        components=1,
        restrictions=[ZeroTrace(component=0, boundary="all")],
    )
    basis = space.basis("laplacian", size=4, degree=1)

    def weak(u, v, dx, ds):
        return -0.05 * inner(grad(u[0]), grad(v[0])) * dx

    problem = NeuralSemigroupProblem(basis=basis, weak=weak, radius=1.5)
    semigroup = problem.train(
        hidden=(64, 64),
        lipschitz=10.0,
        samples=4096,
        batch_size=256,
        epochs=500,
        lr=1e-3,
        jacobian_weight=0.1,
        balance_epsilon=1e-6,
        seed=0,
        verbose=True,
    )
    z0 = semigroup.project(lambda x: torch.sin(torch.pi * x[:, :1]))
    times = torch.linspace(0, 0.2, 21, dtype=semigroup.dtype, device=semigroup.device)
    states = semigroup.solve(z0, times)
    print(semigroup)
    print(dict(semigroup.metrics))
    print(states.shape)


if __name__ == "__main__":
    main()
