import unittest

import torch
from ngfield import DomainExitError, integrate_field

from galerkin_neural_semigroup._network import _LocalBallField, _SpectralMLP
from galerkin_neural_semigroup.semigroup import NeuralSemigroup

from ._fixtures import heat_problem


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.problem = heat_problem(radius=3.0, sobolev_order=1)
        coordinate_system = self.problem._build_coordinate_system(
            device=torch.device("cpu"), dtype=torch.float64
        )
        core = _SpectralMLP(2, (), 1.0, device=torch.device("cpu"), dtype=torch.float64)
        with torch.no_grad():
            core.weights[0].copy_(-0.5 * torch.eye(2, dtype=torch.float64))
            core.biases[0].zero_()
        field = _LocalBallField(core, radius=3.0, sobolev_order=1)
        self.semigroup = NeuralSemigroup(
            field=field, coordinate_system=coordinate_system, radius=3.0
        )

    def test_physical_time_forward_backward_and_defect(self):
        state = torch.tensor([1.0, -2.0], dtype=torch.float64)
        result = self.semigroup(state, 0.8, tolerance=1e-11)
        expected = state * torch.exp(torch.tensor(-0.4, dtype=torch.float64))
        torch.testing.assert_close(result, expected, atol=2e-9, rtol=2e-9)
        recovered = self.semigroup(result, -0.8, tolerance=1e-11)
        torch.testing.assert_close(recovered, state, atol=5e-9, rtol=5e-9)
        torch.testing.assert_close(self.semigroup.velocity(state), -0.5 * state)
        self.assertLess(
            torch.linalg.vector_norm(
                self.semigroup.defect(state, 0.2, 0.3, tolerance=1e-11)
            ).item(),
            1e-9,
        )

    def test_both_fields_use_the_ngf_integrator_with_identical_controls(self):
        state = torch.tensor([1.0, 0.5], dtype=torch.float64)
        times = torch.linspace(0, 0.2, 3, dtype=torch.float64)
        learned = self.semigroup.solve(state, times, step=0.02, order=3)
        direct_learned = integrate_field(
            self.semigroup.field, state, times, step=0.02, order=3, radius=3.0
        )
        torch.testing.assert_close(learned, direct_learned)
        galerkin = self.problem._build_coordinate_system(
            device=torch.device("cpu"), dtype=torch.float64
        )
        torch.testing.assert_close(
            galerkin.solve(state, times, step=0.02, order=3, radius=3.0),
            integrate_field(galerkin, state, times, step=0.02, order=3, radius=3.0),
        )
        points = torch.linspace(0, 1, 5, dtype=torch.float64).reshape(-1, 1)
        self.assertEqual(self.semigroup.reconstruct(learned, points).shape, (3, 5, 1))

    def test_batch_axes_and_initial_state_gradient(self):
        states = torch.ones(2, 3, 2, dtype=torch.float64) * 0.3
        times = torch.tensor([0.0, 0.1], dtype=torch.float64)
        self.assertEqual(self.semigroup.solve(states, times).shape, (2, 2, 3, 2))
        state = torch.tensor([0.6, -0.2], dtype=torch.float64, requires_grad=True)
        self.semigroup(state, 0.1, tolerance=1e-10).sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_boundary_is_undefined_and_no_exterior_evaluation(self):
        outside = torch.tensor([3.0, 0.0], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "open ball"):
            self.semigroup.field(outside)
        with self.assertRaisesRegex(ValueError, "open ball"):
            self.semigroup(outside, 0.0)
        core = self.semigroup.field.core
        core.train()
        with torch.no_grad():
            core.weights[0].zero_()
            core.biases[0].copy_(torch.tensor([1.0, 0.0], dtype=torch.float64))
        self.semigroup.field.eval()
        with self.assertRaises(DomainExitError):
            self.semigroup.solve(
                torch.tensor([2.9, 0.0], dtype=torch.float64),
                torch.tensor([0.0, 0.2], dtype=torch.float64),
                step=0.05,
            )


if __name__ == "__main__":
    unittest.main()
