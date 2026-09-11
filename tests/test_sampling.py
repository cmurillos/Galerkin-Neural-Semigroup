import unittest

import torch

from galerkin_neural_semigroup._sampling import reference_targets, sample_ball


class FixedSamplingTests(unittest.TestCase):
    def _sample(self, mode, *, count=20_000, dimension=5, radius=2.0):
        return sample_ball(
            count,
            dimension,
            radius,
            sampling=mode,
            generator=torch.Generator().manual_seed(7),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

    def test_volume_sampling_has_normalized_lebesgue_radial_law(self):
        dimension = 5
        radius = 2.0
        states = self._sample("volume", dimension=dimension, radius=radius)
        radii = torch.linalg.vector_norm(states, dim=-1)

        self.assertEqual(states.shape, (20_000, dimension))
        self.assertEqual(states.dtype, torch.float64)
        self.assertTrue(torch.all(radii <= radius))
        self.assertLess(abs((radii / radius).pow(dimension).mean().item() - 0.5), 0.01)

    def test_radius_sampling_has_uniform_radial_law(self):
        radius = 2.0
        states = self._sample("radius", radius=radius)
        radii = torch.linalg.vector_norm(states, dim=-1)

        self.assertTrue(torch.all(radii <= radius))
        self.assertLess(abs((radii / radius).mean().item() - 0.5), 0.01)

    def test_reference_targets_are_scaled_batched_and_detached(self):
        states = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
        calls = []

        def reference(values):
            calls.append(len(values))
            return values.square()

        targets = reference_targets(
            reference,
            states,
            time_scale=0.5,
            batch_size=4,
        )

        self.assertEqual(calls, [4, 4, 3])
        torch.testing.assert_close(targets, 0.5 * states.detach().square())
        self.assertFalse(targets.requires_grad)


if __name__ == "__main__":
    unittest.main()
