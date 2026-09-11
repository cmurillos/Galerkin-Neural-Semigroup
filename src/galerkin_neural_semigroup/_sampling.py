"""Fixed vectorized sampling and private target preparation."""

import torch

SAMPLING_MODES = ("volume", "radius")


def sampling_mode(value):
    """Validate one of the two fixed sampling measures."""
    if not isinstance(value, str):
        raise TypeError("sampling must be 'volume' or 'radius'.")
    if value not in SAMPLING_MODES:
        raise ValueError("sampling must be 'volume' or 'radius'.")
    return value


def sample_ball(count, dimension, radius, *, sampling, generator, device, dtype):
    """Sample a ball uniformly in volume or uniformly in radius."""
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
    return radius * radii * directions


@torch.no_grad()
def reference_targets(reference, states, *, time_scale, batch_size):
    """Return detached scaled field values in vectorized batches."""
    outputs = []
    for start in range(0, len(states), batch_size):
        state = states[start : start + batch_size]
        target = time_scale * reference(state)
        if target.shape != state.shape:
            raise ValueError("The internal reference evaluator changed the state shape.")
        if not torch.isfinite(target).all():
            raise FloatingPointError("The internal reference evaluator returned nonfinite data.")
        outputs.append(target.detach())
    return torch.cat(outputs, dim=0)
