import tempfile
import unittest
from pathlib import Path

import torch

from galerkin_neural_semigroup import NeuralSemigroup
from galerkin_neural_semigroup._network import _SpectralMLP
from galerkin_neural_semigroup.problem import _basis_signature, _field_loss

from ._fixtures import heat_problem


class TrainingTests(unittest.TestCase):
    def test_direct_loss_is_mean_over_states_and_sum_over_components(self):
        prediction = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.zeros_like(prediction)

        loss = _field_loss(prediction, target)
        loss.backward()

        self.assertEqual(float(loss.item()), 15.0)
        torch.testing.assert_close(prediction.grad, prediction.detach())

    def test_linear_training_reduces_independent_field_error(self):
        problem = heat_problem(radius=3.0)
        semigroup = problem.train(
            hidden=(),
            lipschitz=2.0,
            samples=128,
            batch_size=32,
            epochs=40,
            lr=2e-2,
            seed=4,
            device="cpu",
        )
        self.assertIsInstance(semigroup, NeuralSemigroup)
        self.assertLess(
            semigroup.metrics["training_loss"],
            semigroup.history["training_loss"][0],
        )
        self.assertTrue(semigroup.metrics["validation_loss"] >= 0)
        self.assertLessEqual(
            semigroup.metrics["effective_lipschitz_bound"],
            2.0 * (1 + 1e-12),
        )
        self.assertEqual(semigroup.metadata["training"]["target_mode"], "cached")
        self.assertEqual(semigroup.metadata["training"]["sampling"], "volume")
        self.assertEqual(semigroup.metadata["method"]["measure"], "normalized-volume-ball")
        self.assertEqual(
            semigroup.metadata["method"]["loss"],
            "mean-squared-euclidean-field-error",
        )
        self.assertEqual(
            semigroup.metadata["method"]["coordinate_normalization"],
            "unit-ball",
        )
        self.assertEqual(semigroup.metadata["method"]["loss_coordinates"], "unit-ball")
        self.assertEqual(semigroup.metadata["training"]["network_radius"], 1.0)
        self.assertTrue(all(parameter.grad is None for parameter in semigroup.field.parameters()))
        states = torch.randn(5, problem.dimension, dtype=semigroup.dtype)
        torch.testing.assert_close(
            semigroup.field(states),
            problem.radius * semigroup.field.normalized(states / problem.radius),
        )
        self.assertEqual(
            set(semigroup.history),
            {"training_loss", "validation_loss"},
        )
        self.assertIn("quadrature_order", semigroup.metadata["reference"])
        self.assertEqual(semigroup.metadata["method"]["activation"], "tanh")

    def test_radius_sampling_trains_with_the_same_direct_loss(self):
        semigroup = heat_problem().train(
            hidden=(),
            lipschitz=2.0,
            samples=32,
            sampling="radius",
            batch_size=16,
            epochs=2,
            lr=1e-2,
            seed=5,
            device="cpu",
        )

        self.assertEqual(semigroup.metadata["training"]["sampling"], "radius")
        self.assertEqual(semigroup.metadata["method"]["measure"], "uniform-radius-ball")
        self.assertEqual(
            semigroup.metadata["method"]["loss"],
            "mean-squared-euclidean-field-error",
        )

    def test_checkpoint_round_trip_uses_same_problem(self):
        problem = heat_problem()
        semigroup = problem.train(
            hidden=(4,),
            lipschitz=1.5,
            samples=32,
            batch_size=16,
            epochs=3,
            lr=1e-2,
            seed=1,
            device="cpu",
        )
        states = torch.randn(5, problem.dimension, dtype=semigroup.dtype)
        expected = semigroup.field(states)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gns"
            semigroup.save(path)
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint["schema_version"], 2)
            self.assertEqual(
                checkpoint["field_configuration"]["coordinate_normalization"],
                "unit-ball",
            )
            restored = problem.load(path, device="cpu")
        torch.testing.assert_close(restored.field(states), expected)
        self.assertEqual(dict(restored.metrics), dict(semigroup.metrics))
        self.assertEqual(dict(restored.history), dict(semigroup.history))

    def test_schema_one_checkpoint_remains_loadable_without_reinterpretation(self):
        problem = heat_problem(radius=2.0)
        coordinate_system = problem._build_coordinate_system(
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        field = _SpectralMLP(
            problem.dimension,
            (),
            2.0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        field.eval()
        semigroup = NeuralSemigroup(
            field=field,
            coordinate_system=coordinate_system,
            radius=problem.radius,
            time_scale=problem.time_scale,
            metadata={
                "basis": _basis_signature(problem.basis),
                "dtype": "float64",
            },
        )
        states = torch.randn(4, problem.dimension, dtype=torch.float64)
        expected = semigroup.field(states)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.gns"
            semigroup.save(path)
            self.assertEqual(torch.load(path, weights_only=True)["schema_version"], 1)
            restored = problem.load(path, device="cpu")

        torch.testing.assert_close(restored.field(states), expected)
        self.assertFalse(hasattr(restored.field, "normalized"))

    def test_load_rejects_different_training_domain(self):
        problem = heat_problem(radius=1.0)
        semigroup = problem.train(
            hidden=(),
            lipschitz=2.0,
            samples=16,
            batch_size=8,
            epochs=1,
            lr=1e-2,
            seed=0,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gns"
            semigroup.save(path)
            incompatible = heat_problem(radius=2.0)
            with self.assertRaisesRegex(ValueError, "radii"):
                incompatible.load(path, device="cpu")


if __name__ == "__main__":
    unittest.main()
