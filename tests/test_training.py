import tempfile
import unittest
from pathlib import Path

import torch

from galerkin_neural_semigroup import NeuralSemigroup
from galerkin_neural_semigroup._network import _SpectralMLP, _UnitBallField
from galerkin_neural_semigroup.problem import _basis_signature, _field_loss

from ._fixtures import heat_problem


class Quadratic(torch.nn.Module):
    dimension = 1

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def _states(self, states):
        assert states.shape[-1] == 1

    def forward(self, states):
        return self.weight * states.square()


class TrainingTests(unittest.TestCase):
    def test_taylor_order_is_independent_of_training_sobolev_order(self):
        model = heat_problem(sobolev_order=0).train(
            hidden=(), lipschitz=2.0, samples=8, batch_size=4, epochs=1, device="cpu"
        )
        state = torch.zeros(2, dtype=model.dtype)
        times = torch.tensor([0.0, 0.1], dtype=model.dtype)
        self.assertEqual(model.solve(state, times, order=3).shape, (2, 2))
        self.assertEqual(model.metadata["method"]["sobolev_order"], 0)

    def test_h2_loss_sums_each_indexed_derivative_then_averages_states(self):
        field = Quadratic()
        states = torch.tensor([[0.25], [0.5]], dtype=torch.float64)
        targets = {
            (0,): torch.zeros_like(states),
            (1,): torch.zeros_like(states),
            (2,): torch.zeros_like(states),
        }
        loss = _field_loss(field, states, targets, 2)
        expected = (states.pow(4) + 4 * states.square() + 4).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(field.weight.grad)
        self.assertGreater(field.weight.grad.item(), 0)

    def test_sobolev_training_and_schema_five_round_trip(self):
        problem = heat_problem(radius=3.0, sobolev_order=2)
        semigroup = problem.train(
            hidden=(),
            lipschitz=2.0,
            samples=64,
            batch_size=16,
            epochs=12,
            lr=2e-2,
            seed=4,
            device="cpu",
        )
        self.assertLess(semigroup.metrics["training_loss"], semigroup.history["training_loss"][0])
        self.assertEqual(semigroup.metadata["method"]["sobolev_order"], 2)
        self.assertEqual(semigroup.metadata["method"]["loss"], "mean-squared-indexed-sobolev-error")
        self.assertEqual(semigroup.metadata["training"]["sampling"], "volume")
        self.assertEqual(
            semigroup.metadata["reference_package"], "numerical-galerkin-field@a466f135"
        )
        self.assertLessEqual(semigroup.metrics["effective_lipschitz_bound"], 2.0 * (1 + 1e-12))
        states = torch.tensor([[0.3, 0.2], [-0.1, 0.4]], dtype=semigroup.dtype)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gns"
            semigroup.save(path)
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint["schema_version"], 5)
            self.assertNotIn("time_scale", checkpoint)
            self.assertEqual(checkpoint["field_configuration"]["domain"], "open-ball")
            restored = problem.load(path, device="cpu")
            torch.testing.assert_close(restored.field(states), semigroup.field(states))
            with self.assertRaisesRegex(ValueError, "Sobolev orders"):
                heat_problem(radius=3.0, sobolev_order=1).load(path, device="cpu")
            with self.assertRaisesRegex(ValueError, "radii"):
                heat_problem(radius=2.0, sobolev_order=2).load(path, device="cpu")
        self.assertEqual(dict(restored.metrics), dict(semigroup.metrics))

    def test_automatic_calibration_uses_indexed_reference_derivatives(self):
        semigroup = heat_problem(radius=1.5).train(
            hidden=(), samples=16, batch_size=8, epochs=1, seed=7, device="cpu"
        )
        calibration = semigroup.metadata["training"]["lipschitz_calibration"]
        self.assertEqual(calibration["mode"], "empirical-indexed-derivatives")
        self.assertFalse(calibration["certified_upper_bound"])
        self.assertGreater(calibration["reference_estimate_normalized"], 0)
        self.assertAlmostEqual(
            calibration["network_budget"], 1.5 * calibration["reference_estimate_normalized"]
        )

    def test_legacy_schemas_one_through_four_keep_fields_and_time_factor(self):
        problem = heat_problem(radius=2.0)
        system = problem._build_coordinate_system(device=torch.device("cpu"), dtype=torch.float64)
        state = torch.tensor([2.5, 0.0], dtype=torch.float64)
        for schema in (1, 2, 3, 4):
            with self.subTest(schema=schema):
                core = _SpectralMLP(2, (), 2.0, device=torch.device("cpu"), dtype=torch.float64)
                with torch.no_grad():
                    core.weights[0].zero_()
                    core.biases[0].fill_(1.0)
                if schema == 1:
                    field = core
                else:
                    field = _UnitBallField(
                        core,
                        2.0,
                        compact_support=schema in (3, 4),
                        support_start_radius=0.9 if schema == 3 else 1.0,
                        support_end_radius=1.0 if schema == 3 else 2.0,
                    )
                model = NeuralSemigroup(
                    field=field,
                    coordinate_system=system,
                    radius=2.0,
                    time_scale=2.0,
                    metadata={"basis": _basis_signature(problem.basis), "dtype": "float64"},
                )
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "legacy.gns"
                    model.save(path)
                    self.assertEqual(torch.load(path, weights_only=True)["schema_version"], schema)
                    restored = problem.load(path, device="cpu")
                torch.testing.assert_close(restored.field(state), model.field(state))
                torch.testing.assert_close(restored.velocity(state), model.field(state) / 2)


if __name__ == "__main__":
    unittest.main()
