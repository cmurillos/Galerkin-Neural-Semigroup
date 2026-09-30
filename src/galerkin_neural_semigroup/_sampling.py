"""Fixed uniform-volume sampling and indexed Sobolev targets on the unit ball."""

import torch
from ngfield import multi_indices


def sample_unit_ball(count, dimension, *, generator, device, dtype):
    """Sample normalized Lebesgue volume on the open unit ball."""
    directions = torch.randn(count, dimension, generator=generator, device=device, dtype=dtype)
    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(
        torch.finfo(dtype).tiny
    )
    radii = torch.rand(count, 1, generator=generator, device=device, dtype=dtype)
    return directions * radii.pow(1.0 / dimension)


def unit_ball_targets(reference, states, *, radius, order, batch_size):
    """Cache all derivatives of ``g_R(x)=G(R*x)/R`` up to ``order``.

    For multi-index ``alpha`` the normalized derivative is
    ``R^(|alpha|-1) * partial_z^alpha G(R*x)``. Detachment follows
    differentiation, so the reference never enters the optimizer.
    """
    indices = multi_indices(states.shape[-1], order)
    outputs = {alpha: torch.empty_like(states) for alpha in indices}
    for start in range(0, len(states), batch_size):
        state = states[start : start + batch_size]
        derivatives = reference.state_derivatives(radius * state, order)
        for alpha in indices:
            value = (radius ** (sum(alpha) - 1) * derivatives[alpha]).detach()
            if value.shape != state.shape:
                raise ValueError("The reference derivative changed the state shape.")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("The reference derivative returned nonfinite data.")
            outputs[alpha][start : start + len(state)].copy_(value)
    return outputs
