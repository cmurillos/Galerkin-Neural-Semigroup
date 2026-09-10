import unittest

import numpy as np
from ngfield import GalerkinProblem

from galerkin_neural_semigroup import NeuralSemigroupProblem

from ._fixtures import heat_problem


class ProblemContractTests(unittest.TestCase):
    def test_normal_route_exposes_no_reference_field(self):
        problem = heat_problem()
        self.assertEqual(problem.dimension, 2)
        self.assertFalse(hasattr(problem, "field"))
        self.assertFalse(hasattr(problem, "reference"))
        self.assertIn("dimension=2", repr(problem))

    def test_explicit_galerkin_problem_adapter(self):
        vertices = np.array([[0.0], [0.5], [1.0]])
        simplices = np.array([[0, 1], [1, 2]])

        def weak(u, v, dx, ds):
            return -u * v * dx

        legacy = GalerkinProblem(vertices=vertices, simplices=simplices, weak=weak)
        basis = legacy.basis("laplacian", size=2, degree=1)
        problem = NeuralSemigroupProblem.from_galerkin(
            legacy,
            basis=basis,
            radius=2.0,
        )
        self.assertEqual(problem.dimension, 2)
        self.assertEqual(problem.radius, 2.0)
        semigroup = problem.train(
            hidden=(),
            lipschitz=2.0,
            samples=8,
            batch_size=4,
            epochs=1,
            lr=1e-2,
            device="cpu",
        )
        self.assertEqual(semigroup.dimension, 2)

    def test_invalid_public_inputs_are_rejected(self):
        valid = heat_problem()
        with self.assertRaisesRegex(ValueError, "radius"):
            NeuralSemigroupProblem(basis=valid.basis, weak=valid.weak, radius=0)
        with self.assertRaisesRegex(TypeError, "weak"):
            NeuralSemigroupProblem(basis=valid.basis, weak=None, radius=1)
        with self.assertRaisesRegex(ValueError, "time_scale"):
            NeuralSemigroupProblem(
                basis=valid.basis,
                weak=valid.weak,
                radius=1,
                time_scale=-1,
            )
        training = {
            "hidden": (),
            "lipschitz": 2.0,
            "samples": 2,
            "batch_size": 2,
            "epochs": 1,
            "lr": 1e-2,
            "device": "cpu",
        }
        with self.assertRaisesRegex(ValueError, "jacobian_weight"):
            valid.train(**training, jacobian_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "balance_epsilon"):
            valid.train(**training, balance_epsilon=0)


if __name__ == "__main__":
    unittest.main()
