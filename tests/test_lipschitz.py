import unittest

import torch
from ngfield import state_derivatives

from galerkin_neural_semigroup._lipschitz import estimate_reference_lipschitz


class Reference:
    dimension = 2

    def __init__(self, matrix, offset=None):
        self.matrix = matrix
        self.offset = offset

    def _states(self, states):
        assert states.shape[-1] == 2

    def __call__(self, states):
        result = states @ self.matrix.T
        return result if self.offset is None else result + self.offset

    def state_derivatives(self, states, order):
        return state_derivatives(self, states, order)


class ReferenceLipschitzTests(unittest.TestCase):
    def test_affine_offset_does_not_change_sampled_derivative_norm(self):
        matrix = torch.diag(torch.tensor([3.0, 0.4], dtype=torch.float64))
        offset = torch.tensor([12.0, -5.0], dtype=torch.float64)
        estimate, probes = estimate_reference_lipschitz(
            Reference(matrix, offset),
            2,
            radius=2.0,
            batch_size=17,
            generator=torch.Generator().manual_seed(91),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        self.assertAlmostEqual(estimate, 3.0, places=12)
        self.assertGreaterEqual(probes, 16)

    def test_reproducibility_and_float32(self):
        matrix = torch.diag(torch.tensor([2.0, 0.1], dtype=torch.float32))
        args = dict(radius=1.0, batch_size=16, device=torch.device("cpu"), dtype=torch.float32)
        first = estimate_reference_lipschitz(
            Reference(matrix), 2, generator=torch.Generator().manual_seed(7), **args
        )
        second = estimate_reference_lipschitz(
            Reference(matrix), 2, generator=torch.Generator().manual_seed(7), **args
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
