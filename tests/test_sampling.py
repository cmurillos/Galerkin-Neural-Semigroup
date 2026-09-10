import unittest

import torch

from galerkin_neural_semigroup._sampling import (
    adaptive_refinement,
    canonical_directions,
    initial_radial_layers,
    radial_design,
    radial_error_profile,
    radial_interval_probe,
    reference_targets,
    reference_targets_and_jacobians,
)


class RadialSamplingTests(unittest.TestCase):
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

    def test_design_uses_exact_layers_and_normalized_sphere_directions(self):
        dimension = 5
        radius = 2.0
        generator = torch.Generator().manual_seed(7)
        layers = initial_radial_layers(
            2_000,
            dimension,
            radius,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        states, layer_indices = radial_design(
            2_000,
            dimension,
            layers,
            generator=generator,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        radii = torch.linalg.vector_norm(states, dim=-1)
        torch.testing.assert_close(radii, layers[layer_indices])
        self.assertEqual(int((radii == 0).sum().item()), 1)
        self.assertEqual(float(layers[0].item()), 0.0)
        self.assertEqual(float(layers[-1].item()), radius)
        counts = torch.bincount(layer_indices, minlength=len(layers))[1:]
        self.assertLessEqual(int(counts.max().item() - counts.min().item()), 1)

    def test_radial_profile_averages_each_spherical_layer(self):
        scores = torch.tensor([2.0, 4.0, 8.0, 12.0], dtype=torch.float64)
        indices = torch.tensor([0, 1, 1, 2])
        layers = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

        profile = radial_error_profile(scores, indices, layers)

        torch.testing.assert_close(
            profile,
            torch.tensor([2.0, 6.0, 12.0], dtype=torch.float64),
        )

    def test_probe_samples_every_adjacent_layer_midpoint(self):
        generator = torch.Generator().manual_seed(11)
        layers = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64)

        states, interval_indices = radial_interval_probe(
            128,
            3,
            layers,
            generator=generator,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        midpoints = torch.tensor([0.125, 0.625], dtype=torch.float64)
        torch.testing.assert_close(
            torch.linalg.vector_norm(states, dim=-1),
            midpoints[interval_indices],
        )
        counts = torch.bincount(interval_indices, minlength=2)
        self.assertLessEqual(int(counts.max().item() - counts.min().item()), 1)

    def test_adaptive_refinement_inserts_worst_interval_midpoint_layer(self):
        generator = torch.Generator().manual_seed(5)
        layers = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        candidates, interval_indices = radial_interval_probe(
            128,
            2,
            layers,
            generator=generator,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        scores = torch.where(
            interval_indices == 0,
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(10.0, dtype=torch.float64),
        )

        states, new_radius, radial_errors, interval_index = adaptive_refinement(
            candidates,
            scores,
            interval_indices,
            layers,
            512,
            generator=generator,
        )

        self.assertEqual(interval_index, 1)
        self.assertEqual(float(new_radius.item()), 0.75)
        torch.testing.assert_close(
            radial_errors,
            torch.tensor([1.0, 10.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(states, dim=-1),
            torch.full((512,), 0.75, dtype=torch.float64),
        )


if __name__ == "__main__":
    unittest.main()
