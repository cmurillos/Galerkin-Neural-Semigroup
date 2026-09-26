import unittest

import torch

from galerkin_neural_semigroup._lipschitz import estimate_reference_lipschitz


class ReferenceLipschitzTests(unittest.TestCase):
    def test_affine_field_recovers_spectral_norm_with_offset_and_time_scale(self):
        matrix = torch.diag(torch.tensor([3.0, 0.4], dtype=torch.float64))
        offset = torch.tensor([12.0, -5.0], dtype=torch.float64)

        def field(states):
            return states @ matrix.T + offset

        estimate, probes, step = estimate_reference_lipschitz(
            field,
            2,
            radius=2.0,
            time_scale=0.25,
            batch_size=17,
            generator=torch.Generator().manual_seed(91),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        self.assertAlmostEqual(estimate, 0.75, places=9)
        self.assertGreaterEqual(probes, 16)
        self.assertLess(step, 0.01)

    def test_nonlinear_estimate_is_reproducible_but_not_a_certificate(self):
        def field(states):
            return states.square()

        def estimate():
            return estimate_reference_lipschitz(
                field,
                2,
                radius=1.0,
                time_scale=1.0,
                batch_size=32,
                generator=torch.Generator().manual_seed(7),
                device=torch.device("cpu"),
                dtype=torch.float64,
            )

        first = estimate()
        self.assertEqual(first, estimate())
        self.assertLess(first[0], 2.0)
        self.assertGreater(first[0], 0.0)

    def test_float32_uses_a_stable_difference_step(self):
        matrix = torch.diag(torch.tensor([2.0, 0.1], dtype=torch.float32))

        def field(states):
            return states @ matrix.T

        estimate, _, step = estimate_reference_lipschitz(
            field,
            2,
            radius=1.0,
            time_scale=1.0,
            batch_size=32,
            generator=torch.Generator().manual_seed(11),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertAlmostEqual(estimate, 2.0, places=4)
        self.assertEqual(step, 1e-2)


if __name__ == "__main__":
    unittest.main()
