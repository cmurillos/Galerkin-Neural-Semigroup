import unittest

import torch

from galerkin_neural_semigroup._network import _SpectralMLP, _UnitBallField


class SpectralMLPTests(unittest.TestCase):
    def test_shape_gradient_and_global_budget(self):
        field = _SpectralMLP(
            4,
            (7, 5),
            0.4,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        states = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        values = field(states)
        self.assertEqual(values.shape, states.shape)
        values.square().sum().backward()
        self.assertTrue(torch.isfinite(states.grad).all())
        self.assertTrue(all(parameter.grad is not None for parameter in field.parameters()))
        self.assertTrue(
            all(torch.isfinite(parameter.grad).all() for parameter in field.parameters())
        )
        self.assertLessEqual(field.effective_lipschitz_bound(), 0.4 * (1 + 1e-12))
        for norm in field.spectral_norms():
            self.assertLessEqual(norm, field.layer_bound * (1 + 1e-12))

    def test_empty_hidden_selects_an_affine_field(self):
        field = _SpectralMLP(
            2,
            (),
            3.0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        with torch.no_grad():
            field.weights[0].copy_(torch.tensor([[-0.5, 0.0], [0.0, -0.25]]))
            field.biases[0].zero_()
        state = torch.tensor([2.0, 4.0], dtype=torch.float64)
        torch.testing.assert_close(field(state), torch.tensor([-1.0, -1.0], dtype=torch.float64))

    def test_evaluation_caches_the_fixed_projected_weights(self):
        field = _SpectralMLP(
            3,
            (5,),
            0.7,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        field.eval()
        first_weights = field.effective_weights()
        first_biases = field.effective_biases()
        self.assertIs(first_weights, field.effective_weights())
        self.assertIs(first_biases, field.effective_biases())
        field.train()
        self.assertIsNone(field._evaluation_weights)
        self.assertIsNone(field._evaluation_biases)

    def test_evaluation_tracks_states_only_when_requested(self):
        field = _SpectralMLP(
            3,
            (5,),
            0.7,
            device=torch.device("cpu"),
            dtype=torch.float64,
        ).eval()

        ordinary = torch.randn(4, 3, dtype=torch.float64)
        self.assertFalse(field(ordinary).requires_grad)

        differentiable = ordinary.clone().requires_grad_()
        field(differentiable).sum().backward()
        self.assertTrue(torch.isfinite(differentiable.grad).all())
        self.assertTrue(all(parameter.grad is None for parameter in field.parameters()))

    def test_state_contract_is_strict(self):
        field = _SpectralMLP(
            3,
            (4,),
            1.0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            field(torch.zeros(2, 4, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "device and dtype"):
            field(torch.zeros(2, 3, dtype=torch.float32))
        with self.assertRaises(TypeError):
            field(torch.zeros(3, dtype=torch.float64), 0.2)

    def test_unit_ball_field_preserves_physical_coordinates(self):
        core = _SpectralMLP(
            2,
            (),
            3.0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        with torch.no_grad():
            core.weights[0].copy_(torch.tensor([[-0.5, 0.0], [0.0, -0.25]]))
            core.biases[0].copy_(torch.tensor([0.1, -0.2], dtype=torch.float64))
        field = _UnitBallField(core, radius=4.0)
        states = torch.tensor([[2.0, -4.0], [-1.0, 3.0]], dtype=torch.float64)

        torch.testing.assert_close(field(states), 4.0 * core(states / 4.0))
        torch.testing.assert_close(field.normalized(states / 4.0), core(states / 4.0))
        self.assertEqual(field.configuration()["coordinate_normalization"], "unit-ball")
        self.assertEqual(field.effective_lipschitz_bound(), core.effective_lipschitz_bound())

        field.eval()
        field(states)
        zero_state = {name: torch.zeros_like(value) for name, value in field.state_dict().items()}
        field.load_state_dict(zero_state)
        torch.testing.assert_close(field(states), torch.zeros_like(states))


if __name__ == "__main__":
    unittest.main()
