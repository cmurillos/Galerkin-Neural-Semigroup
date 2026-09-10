import unittest

import torch

from galerkin_neural_semigroup._loss import (
    adaptive_field_score,
    component_weights,
    field_loss,
    field_loss_terms,
)


class FieldLossTests(unittest.TestCase):
    def test_component_weights_are_inverse_regularized_rms(self):
        targets = torch.tensor([[1.0, 4.0], [-1.0, 0.0]], dtype=torch.float64)
        weights = component_weights(targets, epsilon=0.1)
        energy = targets.square().mean(dim=0)
        expected = torch.rsqrt(energy + 0.1 * energy.mean())

        torch.testing.assert_close(weights, expected)

    def test_identically_zero_field_uses_identity_weights(self):
        targets = torch.zeros((4, 3), dtype=torch.float64)

        torch.testing.assert_close(
            component_weights(targets, epsilon=1e-6),
            torch.ones(3, dtype=torch.float64),
        )

    def test_terms_match_the_defining_formula(self):
        prediction = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], dtype=torch.float64)
        target = torch.tensor([[2.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        prediction_jacobian = torch.tensor(
            [[[0.5, -1.0], [2.0, 1.0]], [[1.0, 0.0], [-2.0, 3.0]]],
            dtype=torch.float64,
        )
        target_jacobian = torch.tensor(
            [[[1.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [-1.0, 1.0]]],
            dtype=torch.float64,
        )
        weights = torch.tensor([2.0, 0.5], dtype=torch.float64)

        value, jacobian = field_loss_terms(
            prediction,
            target,
            prediction_jacobian,
            target_jacobian,
            weights=weights,
        )
        expected_value = ((prediction - target) * weights).square().mean(dim=-1)
        expected_jacobian = (
            ((prediction_jacobian - target_jacobian) * weights[None, :, None])
            .square()
            .mean(dim=(-2, -1))
        )

        torch.testing.assert_close(value, expected_value)
        torch.testing.assert_close(jacobian, expected_jacobian)

    def test_zero_fields_have_zero_finite_loss_and_gradient(self):
        prediction = torch.zeros((3, 2), dtype=torch.float64, requires_grad=True)
        target = torch.zeros_like(prediction)
        prediction_jacobian = torch.zeros((3, 2, 2), dtype=torch.float64, requires_grad=True)
        target_jacobian = torch.zeros_like(prediction_jacobian)
        loss = field_loss(
            prediction,
            target,
            prediction_jacobian,
            target_jacobian,
            weights=torch.ones(2, dtype=torch.float64),
            jacobian_weight=0.1,
        )

        loss.backward()
        torch.testing.assert_close(loss, torch.tensor(0.0, dtype=torch.float64))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.isfinite(prediction_jacobian.grad).all())

    def test_refinement_score_uses_independent_value_error(self):
        prediction = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        target = torch.zeros_like(prediction)

        score = adaptive_field_score(
            prediction,
            target,
            weights=torch.ones(2, dtype=torch.float64),
        )

        torch.testing.assert_close(score, torch.tensor([0.5], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
