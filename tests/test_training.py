import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from galerkin_neural_semigroup import NeuralSemigroup

from ._fixtures import heat_problem


class TrainingTests(unittest.TestCase):
    def test_linear_training_reduces_independent_field_error(self):
        problem = heat_problem()
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
        self.assertEqual(semigroup.metadata["training"]["jacobian_weight"], 0.1)
        self.assertEqual(semigroup.metadata["training"]["balance_epsilon"], 1e-6)
        self.assertEqual(
            semigroup.metadata["training"]["jacobian_probe"],
            "complete-radius-scaled-canonical-basis",
        )
        self.assertEqual(semigroup.metadata["training"]["jacobian_directions"], problem.dimension)
        self.assertIn("quadrature_order", semigroup.metadata["reference"])
        self.assertEqual(semigroup.metadata["method"]["activation"], "tanh")
        self.assertEqual(
            semigroup.metadata["method"]["loss"],
            "component-balanced-sobolev-full-jacobian",
        )
        self.assertEqual(
            semigroup.metadata["method"]["validation_objective"],
            "component-balanced-field-values",
        )
        self.assertIn("training_value_loss", semigroup.history)
        self.assertIn("validation_jacobian_loss", semigroup.history)
        self.assertGreaterEqual(len(semigroup.history["validation_jacobian_audits"]), 1)
        self.assertTrue(semigroup.metrics["validation_value_loss"] >= 0)
        self.assertGreaterEqual(semigroup.metrics["validation_jacobian_loss"], 0)
        self.assertAlmostEqual(
            semigroup.metrics["validation_loss"],
            semigroup.metrics["validation_value_loss"],
        )
        self.assertAlmostEqual(
            semigroup.metrics["training_loss"],
            semigroup.metrics["training_value_loss"]
            + 0.1 * semigroup.metrics["training_jacobian_loss"],
        )
        self.assertLess(
            semigroup.metrics["validation_value_loss"],
            semigroup.history["validation_value_loss"][0],
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
            restored = problem.load(path, device="cpu")
        torch.testing.assert_close(restored.field(states), expected)
        self.assertEqual(dict(restored.metrics), dict(semigroup.metrics))
        self.assertEqual(dict(restored.history), dict(semigroup.history))

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

    def test_tolerance_stops_only_after_minimum_epochs(self):
        semigroup = heat_problem().train(
            hidden=(),
            lipschitz=2.0,
            samples=16,
            batch_size=8,
            epochs=2,
            max_epochs=6,
            tolerance=1e12,
            candidate_samples=16,
            patience=1,
            lr=1e-2,
            seed=2,
            device="cpu",
        )

        self.assertEqual(semigroup.metrics["stop_reason"], "tolerance")
        self.assertEqual(semigroup.metrics["epochs_completed"], 2)
        self.assertEqual(len(semigroup.history["candidate_checks"]), 1)

    def test_plateau_with_coverage_gap_adds_adaptive_samples(self):
        with patch(
            "galerkin_neural_semigroup.problem._upper_quantile",
            side_effect=[10.0, 10.0, 1.0, 5.0, 5.0],
        ):
            semigroup = heat_problem().train(
                hidden=(),
                lipschitz=2.0,
                samples=16,
                batch_size=8,
                epochs=1,
                max_epochs=3,
                refine_every=1,
                refine_samples=4,
                candidate_samples=16,
                patience=1,
                lr=1e-12,
                seed=3,
                device="cpu",
            )

        self.assertEqual(semigroup.metrics["refinements"], 1)
        self.assertEqual(semigroup.metrics["training_samples"], 20)
        self.assertEqual(semigroup.metadata["training"]["target_mode"], "cached-and-appended")

    def test_max_epochs_cannot_precede_minimum_epochs(self):
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            heat_problem().train(
                hidden=(),
                lipschitz=2.0,
                epochs=3,
                max_epochs=2,
            )


if __name__ == "__main__":
    unittest.main()
