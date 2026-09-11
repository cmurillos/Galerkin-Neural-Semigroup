import unittest

import torch

from galerkin_neural_semigroup._sampling import sample_unit_ball, unit_ball_targets


class FixedSamplingTests(unittest.TestCase):
    def _sample(self, mode, *, count=20_000, dimension=5):
        return sample_unit_ball(
            count,
            dimension,
            sampling=mode,
            generator=torch.Generator().manual_seed(7),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

    def test_volume_sampling_has_normalized_lebesgue_radial_law(self):
        dimension = 5
        states = self._sample("volume", dimension=dimension)
        radii = torch.linalg.vector_norm(states, dim=-1)

        self.assertEqual(states.shape, (20_000, dimension))
        self.assertEqual(states.dtype, torch.float64)
        self.assertTrue(torch.all(radii <= 1.0))
        self.assertLess(abs(radii.pow(dimension).mean().item() - 0.5), 0.01)

    def test_radius_sampling_has_uniform_radial_law(self):
        states = self._sample("radius")
        radii = torch.linalg.vector_norm(states, dim=-1)

        self.assertTrue(torch.all(radii <= 1.0))
        self.assertLess(abs(radii.mean().item() - 0.5), 0.01)

    def test_targets_use_physical_states_and_normalized_field_scale(self):
        states = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
        calls = []

        def reference(values):
            calls.append(values.detach().clone())
            return values.square()

        targets = unit_ball_targets(
            reference,
            states,
            radius=3.0,
            time_scale=0.5,
            batch_size=4,
        )

        self.assertEqual([len(values) for values in calls], [4, 4, 3])
        torch.testing.assert_close(torch.cat(calls), 3.0 * states.detach())
        torch.testing.assert_close(targets, 1.5 * states.detach().square())
        self.assertFalse(targets.requires_grad)


if __name__ == "__main__":
    unittest.main()
