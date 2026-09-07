import unittest

import torch

from galerkin_neural_semigroup._network import _SpectralMLP
from galerkin_neural_semigroup.semigroup import NeuralSemigroup

from ._fixtures import heat_problem


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.problem = heat_problem(time_scale=2.0)
        coordinate_system = self.problem._build_coordinate_system(
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        field = _SpectralMLP(
            2,
            (),
            1.0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        with torch.no_grad():
            field.weights[0].copy_(-0.5 * torch.eye(2, dtype=torch.float64))
            field.biases[0].zero_()
        self.semigroup = NeuralSemigroup(
            field=field,
            coordinate_system=coordinate_system,
            radius=1.0,
            time_scale=2.0,
        )

    def test_time_scale_and_adaptive_flow(self):
        state = torch.tensor([1.0, -2.0], dtype=torch.float64)
        result = self.semigroup(state, 0.8, tolerance=1e-11)
        expected = state * torch.exp(torch.tensor(-0.2, dtype=torch.float64))
        torch.testing.assert_close(result, expected, atol=2e-10, rtol=2e-10)
        recovered = self.semigroup(result, -0.8, tolerance=1e-11)
        torch.testing.assert_close(recovered, state, atol=4e-10, rtol=4e-10)
        torch.testing.assert_close(self.semigroup.velocity(state), -0.25 * state)

    def test_composition_defect_is_integrator_scale(self):
        state = torch.tensor([0.7, -0.4], dtype=torch.float64)
        defect = self.semigroup.defect(state, 0.2, 0.3, tolerance=1e-11)
        self.assertLess(torch.linalg.vector_norm(defect).item(), 1e-10)

    def test_fixed_rk4_and_physical_reconstruction(self):
        state = torch.tensor([1.0, 0.5], dtype=torch.float64)
        times = torch.linspace(0, 0.2, 3, dtype=torch.float64)
        states = self.semigroup.solve(state, times, step=1e-3)
        self.assertEqual(states.shape, (3, 2))
        points = torch.linspace(0, 1, 5, dtype=torch.float64).reshape(-1, 1)
        physical = self.semigroup.reconstruct(states, points)
        self.assertEqual(physical.shape, (3, 5, 1))

    def test_arbitrary_leading_batch_axes_are_preserved(self):
        states = torch.ones(2, 3, 2, dtype=torch.float64)
        times = torch.tensor([0.0, 0.1], dtype=torch.float64)
        trajectory = self.semigroup.solve(states, times, tolerance=1e-10)
        self.assertEqual(trajectory.shape, (2, 2, 3, 2))

    def test_flow_remains_differentiable_with_respect_to_initial_state(self):
        state = torch.tensor([0.6, -0.2], dtype=torch.float64, requires_grad=True)
        result = self.semigroup(state, 0.1, tolerance=1e-10)
        result.sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())


if __name__ == "__main__":
    unittest.main()
