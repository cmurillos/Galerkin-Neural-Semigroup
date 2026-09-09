import unittest

import torch

from galerkin_neural_semigroup._sampling import (
    adaptive_refinement,
    radial_probe,
    uniform_ball,
)


class UniformBallTests(unittest.TestCase):
    def test_support_and_radial_distribution(self):
        dimension = 5
        radius = 2.0
        generator = torch.Generator().manual_seed(7)
        states = uniform_ball(
            20_000,
            dimension,
            radius,
            generator=generator,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        radii = torch.linalg.vector_norm(states, dim=-1)
        self.assertTrue(torch.all(radii <= radius))
        # Uniform volume has E[(r/R)^N] = 1/2.
        moment = (radii / radius).pow(dimension).mean().item()
        self.assertLess(abs(moment - 0.5), 0.01)

    def test_radial_probe_covers_the_interior_and_boundary_scales(self):
        generator = torch.Generator().manual_seed(3)
        states = radial_probe(
            1_024,
            8,
            2.0,
            generator=generator,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        radii = torch.linalg.vector_norm(states, dim=-1)

        self.assertTrue(torch.all(radii <= 2.0))
        self.assertTrue(torch.any(radii < 0.2))
        self.assertTrue(torch.any(radii > 1.8))

    def test_adaptive_refinement_follows_error_mass_and_stays_in_ball(self):
        generator = torch.Generator().manual_seed(5)
        positive = torch.tensor([0.7, 0.0], dtype=torch.float64) + 0.02 * torch.randn(
            64,
            2,
            generator=generator,
            dtype=torch.float64,
        )
        negative = torch.tensor([-0.7, 0.0], dtype=torch.float64) + 0.02 * torch.randn(
            64,
            2,
            generator=generator,
            dtype=torch.float64,
        )
        candidates = torch.cat((positive, negative))
        scores = torch.cat(
            (
                torch.full((64,), 100.0, dtype=torch.float64),
                torch.ones(64, dtype=torch.float64),
            )
        )

        states = adaptive_refinement(
            candidates,
            scores,
            512,
            1.0,
            generator=generator,
            exploration=0.0,
        )

        self.assertTrue(torch.all(torch.linalg.vector_norm(states, dim=-1) <= 1.0))
        self.assertGreater(float((states[:, 0] > 0).double().mean()), 0.98)


if __name__ == "__main__":
    unittest.main()
