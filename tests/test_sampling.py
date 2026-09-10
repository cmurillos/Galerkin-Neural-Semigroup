import unittest

import torch

from galerkin_neural_semigroup._sampling import (
    adaptive_refinement,
    canonical_directions,
    radial_probe,
    reference_targets,
    reference_targets_and_jacobians,
    uniform_ball,
)


class UniformBallTests(unittest.TestCase):
    def test_jacobian_directions_are_the_complete_scaled_canonical_basis(self):
        directions = canonical_directions(
            4,
            2.5,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        torch.testing.assert_close(directions, 2.5 * torch.eye(4, dtype=torch.float64))

    def test_reference_values_and_full_jacobians_are_scaled_and_detached(self):
        states = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], dtype=torch.float64)
        matrix = torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=torch.float64)
        targets, target_jacobians = reference_targets_and_jacobians(
            lambda z: z @ matrix.T,
            states,
            radius=2.0,
            time_scale=0.5,
            batch_size=1,
        )

        torch.testing.assert_close(targets, 0.5 * states @ matrix.T)
        torch.testing.assert_close(target_jacobians, matrix.expand(len(states), -1, -1))
        self.assertFalse(targets.requires_grad)
        self.assertFalse(target_jacobians.requires_grad)

    def test_reference_values_can_be_prepared_without_jacobians(self):
        states = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], dtype=torch.float64)
        targets = reference_targets(
            lambda z: z.square(),
            states,
            time_scale=0.5,
            batch_size=1,
        )

        torch.testing.assert_close(targets, 0.5 * states.square())
        self.assertFalse(targets.requires_grad)

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
