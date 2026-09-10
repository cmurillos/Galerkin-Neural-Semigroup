import unittest

import torch

from galerkin_neural_semigroup._sampling import (
    adaptive_refinement,
    jacobian_directions,
    radial_probe,
    reference_targets_and_jvps,
    uniform_ball,
)


class UniformBallTests(unittest.TestCase):
    def test_jacobian_directions_are_radius_scaled_rademacher_vectors(self):
        directions = jacobian_directions(
            100,
            4,
            2.5,
            generator=torch.Generator().manual_seed(1),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        self.assertEqual(directions.shape, (100, 4))
        torch.testing.assert_close(directions.abs(), torch.full_like(directions, 2.5))

    def test_reference_values_and_jvps_are_scaled_and_detached(self):
        states = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], dtype=torch.float64)
        directions = torch.tensor([[2.0, -1.0], [1.0, 2.0]], dtype=torch.float64)
        targets, target_jvps = reference_targets_and_jvps(
            lambda z: z.square(),
            states,
            directions,
            time_scale=0.5,
            batch_size=1,
        )

        torch.testing.assert_close(targets, 0.5 * states.square())
        torch.testing.assert_close(target_jvps, states * directions)
        self.assertFalse(targets.requires_grad)
        self.assertFalse(target_jvps.requires_grad)

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
