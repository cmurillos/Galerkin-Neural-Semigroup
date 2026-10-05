import math
from unittest.mock import patch

import pytest
import torch

from galerkin_neural_semigroup._network import _MLP, _LocalBallField
from galerkin_neural_semigroup._sampling import sample_unit_ball


def test_free_affine_weights_and_posteriori_bound_preserve_training_gradients():
    core = _MLP(2, (), device="cpu", dtype=torch.float64)
    with torch.no_grad():
        core.weights[0].copy_(7 * torch.eye(2, dtype=core.dtype))
        core.biases[0].fill_(0.3)
    x = torch.tensor([[0.1, -0.2]], dtype=core.dtype, requires_grad=True)
    y = core(x)
    torch.testing.assert_close(y, 7 * x + 0.3)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 7))
    assert core.weights[0].grad is not None
    assert math.isclose(core.effective_lipschitz_bound(), 7)


def test_zero_convention_never_evaluates_exterior_and_keeps_local_rejection():
    core = _MLP(2, (), device="cpu", dtype=torch.float64)
    field = _LocalBallField(core, 2.0, 0).eval()
    x = torch.tensor(
        [[[0.2, 0.1], [2.0, 0.0]], [[3.0, 0.0], [0.0, 0.0]]], dtype=core.dtype, requires_grad=True
    )
    with patch.object(core, "forward", wraps=core.forward) as call:
        y = field.zero_extended(x)
    assert y.shape == x.shape
    assert all(
        bool(torch.all(torch.linalg.vector_norm(args.args[0], dim=-1) < 1))
        for args in call.call_args_list
    )
    torch.testing.assert_close(y[0, 0], field(x[0, 0]))
    torch.testing.assert_close(y[0, 1], torch.zeros(2, dtype=core.dtype))
    torch.testing.assert_close(y[1, 0], torch.zeros(2, dtype=core.dtype))
    y.sum().backward()
    torch.testing.assert_close(x.grad[0, 1], torch.zeros(2, dtype=core.dtype))
    with pytest.raises(ValueError, match="open ball"):
        field(x)
    assert field.zero_extended(torch.empty(0, 2, 2, dtype=core.dtype)).shape == (0, 2, 2)
    with pytest.raises(ValueError, match="finite"):
        field.zero_extended(torch.full((2,), float("nan"), dtype=core.dtype))


def test_volume_samples_regenerate_boundary_rounding_in_unit_and_physical_ball():
    # In 64 float32 dimensions, radii this close to one round to the boundary.
    generator = torch.Generator().manual_seed(12)
    original = torch.rand
    calls = 0

    def near_boundary(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.full(
                (args[0], 1), 1 - torch.finfo(torch.float32).eps / 2, dtype=torch.float32
            )
        return original(*args, **kwargs)

    with patch("torch.rand", side_effect=near_boundary):
        states = sample_unit_ball(
            100, 64, generator=generator, device="cpu", dtype=torch.float32, radius=3.1
        )
    assert calls > 1
    assert bool((torch.linalg.vector_norm(states, dim=-1) < 1).all())
    assert bool((torch.linalg.vector_norm(3.1 * states, dim=-1) < 3.1).all())
