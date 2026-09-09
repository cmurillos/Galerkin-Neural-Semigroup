import unittest

import torch

from galerkin_neural_semigroup._loss import (
    adaptive_field_score,
    field_loss,
    field_loss_terms,
)


class FieldLossTests(unittest.TestCase):
    def test_terms_match_the_defining_formula(self):
        prediction = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], dtype=torch.float64)
        target = torch.tensor([[2.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        epsilon = 1e-3

        relative, angular = field_loss_terms(
            prediction,
            target,
            epsilon=epsilon,
        )
        residual_squared = (prediction - target).square().sum(dim=-1)
        prediction_squared = prediction.square().sum(dim=-1)
        target_squared = target.square().sum(dim=-1)
        expected_relative = residual_squared / (target_squared + epsilon)
        expected_angular = 1 - (prediction * target).sum(dim=-1) / torch.sqrt(
            (prediction_squared + epsilon) * (target_squared + epsilon)
        )

        torch.testing.assert_close(relative, expected_relative)
        torch.testing.assert_close(angular, expected_angular)

    def test_alignment_is_preferred_to_orthogonality_and_opposition(self):
        target = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        aligned = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        orthogonal = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        opposite = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
        options = {"angular_weight": 0.1, "epsilon": 1e-8}

        aligned_loss = field_loss(aligned, target, **options)
        orthogonal_loss = field_loss(orthogonal, target, **options)
        opposite_loss = field_loss(opposite, target, **options)

        self.assertLess(aligned_loss, orthogonal_loss)
        self.assertLess(orthogonal_loss, opposite_loss)

    def test_zero_fields_have_finite_loss_and_gradient(self):
        prediction = torch.zeros((3, 2), dtype=torch.float64, requires_grad=True)
        target = torch.zeros_like(prediction)
        loss = field_loss(
            prediction,
            target,
            angular_weight=0.1,
            epsilon=1e-8,
        )

        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_refinement_score_turns_off_undefined_stationary_angle(self):
        prediction = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        target = torch.zeros_like(prediction)
        epsilon = 1e-3

        score = adaptive_field_score(
            prediction,
            target,
            angular_weight=10.0,
            epsilon=epsilon,
        )

        torch.testing.assert_close(score, torch.tensor([1 / epsilon], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
