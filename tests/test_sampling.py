import unittest

import torch
from ngfield import state_derivatives

from galerkin_neural_semigroup._sampling import sample_unit_ball, unit_ball_targets


class PolynomialReference:
    dimension = 2

    def _states(self, states):
        assert states.shape[-1] == 2

    def __call__(self, states):
        a, b = states.unbind(-1)
        return torch.stack((a.square() * b, b.square()), dim=-1)

    def state_derivatives(self, states, order):
        return state_derivatives(self, states, order)


class FixedSamplingTests(unittest.TestCase):
    def test_volume_sampling_has_normalized_lebesgue_radial_law(self):
        generator = torch.Generator().manual_seed(7)
        states = sample_unit_ball(
            20_000, 5, generator=generator, device=torch.device("cpu"), dtype=torch.float64
        )
        radii = torch.linalg.vector_norm(states, dim=-1)
        self.assertEqual(states.shape, (20_000, 5))
        self.assertTrue(torch.all(radii < 1))
        self.assertLess(abs(radii.pow(5).mean().item() - 0.5), 0.01)

    def test_indexed_targets_obey_exact_coordinate_scaling_and_detach(self):
        states = torch.tensor([[0.1, 0.2], [-0.3, 0.4]], dtype=torch.float64, requires_grad=True)
        targets = unit_ball_targets(
            PolynomialReference(), states, radius=3.0, order=2, batch_size=1
        )
        a, b = states.detach().unbind(-1)
        expected = {
            (0, 0): torch.stack((9 * a.square() * b, 3 * b.square()), -1),
            (1, 0): torch.stack((18 * a * b, torch.zeros_like(a)), -1),
            (0, 1): torch.stack((9 * a.square(), 6 * b), -1),
            (2, 0): torch.stack((18 * b, torch.zeros_like(a)), -1),
            (1, 1): torch.stack((18 * a, torch.zeros_like(a)), -1),
            (0, 2): torch.stack((torch.zeros_like(a), torch.full_like(a, 6)), -1),
        }
        self.assertEqual(set(targets), set(expected))
        for alpha, value in expected.items():
            torch.testing.assert_close(targets[alpha], value)
            self.assertFalse(targets[alpha].requires_grad)


if __name__ == "__main__":
    unittest.main()
