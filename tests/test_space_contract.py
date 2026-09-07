import unittest

import numpy as np
import torch

from galerkin_neural_semigroup import NeuralSemigroupProblem
from ngfield import MeanZero, Periodic, SimplicialDomain, Space, grad, inner


class SpaceCompatibilityTests(unittest.TestCase):
    def test_components_are_inferred_from_the_basis(self):
        vertices = np.linspace(0, 1, 7)[:, None]
        simplices = np.column_stack((np.arange(6), np.arange(1, 7)))
        geometry = SimplicialDomain(vertices, simplices)
        space = Space(geometry=geometry, components=2)
        basis = space.basis(
            "laplacian",
            component_sizes=(2, 2),
            degree=1,
        )

        def weak(u, v, dx, ds):
            diffusion = -0.02 * inner(grad(u), grad(v))
            coupling = (u[1] - u[0]) * v[0] + (u[0] - u[1]) * v[1]
            return (diffusion + 0.1 * coupling) * dx

        problem = NeuralSemigroupProblem(basis=basis, weak=weak, radius=1.0)
        semigroup = problem.train(
            hidden=(5,),
            lipschitz=3.0,
            samples=12,
            batch_size=6,
            epochs=1,
            lr=1e-2,
            device="cpu",
        )
        self.assertEqual(problem.dimension, 4)
        self.assertIs(semigroup.space, space)
        states = torch.zeros(2, 4, dtype=semigroup.dtype)
        points = torch.tensor([[0.25], [0.75]], dtype=semigroup.dtype)
        self.assertEqual(semigroup.reconstruct(states, points).shape, (2, 2, 2))

    def test_periodicity_and_mean_zero_need_no_repeated_arguments(self):
        cells = 8
        vertices = np.linspace(0, 1, cells + 1)[:, None]
        simplices = np.column_stack((np.arange(cells), np.arange(1, cells + 1)))
        geometry = SimplicialDomain(
            vertices,
            simplices,
            boundaries={"left": [[0]], "right": [[cells]]},
        )
        space = Space(
            geometry=geometry,
            components=1,
            restrictions=[
                Periodic(
                    component=0,
                    boundaries=("left", "right"),
                    vertex_pairs=[(0, cells)],
                ),
                MeanZero(component=0),
            ],
        )
        basis = space.basis("laplacian", size=2, degree=1)

        def weak(u, v, dx, ds):
            return -0.03 * inner(grad(u[0]), grad(v[0])) * dx

        problem = NeuralSemigroupProblem(basis=basis, weak=weak, radius=1.0)
        semigroup = problem.train(
            hidden=(),
            lipschitz=3.0,
            samples=12,
            batch_size=6,
            epochs=1,
            lr=1e-2,
            device="cpu",
        )
        self.assertEqual(semigroup.dimension, 2)
        self.assertEqual(len(semigroup.space.restrictions), 2)


if __name__ == "__main__":
    unittest.main()
