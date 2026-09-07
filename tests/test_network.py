import unittest

import torch

from galerkin_neural_semigroup._network import _SpectralMLP


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
        first = field.effective_weights()
        second = field.effective_weights()
        self.assertIs(first, second)
        field.train()
        self.assertIsNone(field._evaluation_weights)

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


if __name__ == "__main__":
    unittest.main()
