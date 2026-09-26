"""Fixed vectorized sampling and private target preparation."""

import torch

SAMPLING_MODES = ("volume", "radius", "mixed")


def sampling_mode(value):
    """Validate one of the three fixed sampling measures."""
    if not isinstance(value, str):
        raise TypeError("sampling must be 'volume', 'radius', or 'mixed'.")
    if value not in SAMPLING_MODES:
        raise ValueError("sampling must be 'volume', 'radius', or 'mixed'.")
    return value


def sample_unit_ball(count, dimension, *, sampling, generator, device, dtype):
    """Sample volume, uniform radius, or an independent 50/50 mixture."""
    sampling = sampling_mode(sampling)
    directions = torch.randn(
        count,
        dimension,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    directions = directions / norms.clamp_min(torch.finfo(dtype).tiny)

    radii = torch.rand(
        count,
        1,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if sampling == "volume":
        radii = radii.pow(1.0 / dimension)
    elif sampling == "mixed":
        volume = torch.rand(count, 1, generator=generator, device=device, dtype=dtype) < 0.5
        radii = torch.where(volume, radii.pow(1.0 / dimension), radii)
    return radii * directions


@torch.no_grad()
def unit_ball_targets(reference, states, *, radius, time_scale, batch_size):
    """Evaluate ``(time_scale / radius) * G(radius * x)`` in batches."""
    outputs = torch.empty_like(states, requires_grad=False)
    finite = torch.ones((), device=states.device, dtype=torch.bool)
    for start in range(0, len(states), batch_size):
        state = states[start : start + batch_size]
        target = (time_scale / radius) * reference(radius * state)
        if target.shape != state.shape:
            raise ValueError("The internal reference evaluator changed the state shape.")
        finite.logical_and_(torch.isfinite(target).all())
        outputs[start : start + len(state)].copy_(target)
    if not bool(finite.item()):
        raise FloatingPointError("The internal reference evaluator returned nonfinite data.")
    return outputs
