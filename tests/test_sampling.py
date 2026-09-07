import unittest

import torch

from galerkin_neural_semigroup._sampling import uniform_ball


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


if __name__ == "__main__":
    unittest.main()
