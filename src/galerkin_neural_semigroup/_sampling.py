"""Sampling and target preparation fixed by the learning functional."""

from math import ceil, sqrt

import torch


def uniform_ball(count, dimension, radius, *, generator, device, dtype):
    directions = torch.randn(
        count,
        dimension,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    while bool(torch.any(norms == 0).item()):
        mask = (norms == 0).squeeze(-1)
        directions[mask] = torch.randn(
            int(mask.sum().item()),
            dimension,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    radial = torch.rand(count, 1, generator=generator, device=device, dtype=dtype)
    radial = radius * radial.pow(1.0 / dimension)
    return radial * directions / norms


def radial_probe(count, dimension, radius, *, generator, device, dtype):
    """Return a direction-randomized, radially stratified probe of a ball."""
    levels = min(count, max(2, min(32, round(sqrt(count / dimension)))))
    direction_count = ceil(count / levels)
    directions = torch.randn(
        direction_count,
        dimension,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    while bool(torch.any(norms == 0).item()):
        mask = (norms == 0).squeeze(-1)
        directions[mask] = torch.randn(
            int(mask.sum().item()),
            dimension,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    directions /= norms
    offsets = torch.rand(
        direction_count,
        levels,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    strata = torch.arange(levels, device=device, dtype=dtype)
    radii = radius * (strata[None, :] + offsets) / levels
    states = radii[..., None] * directions[:, None, :]
    return states.reshape(-1, dimension)[:count]


def adaptive_refinement(
    candidates,
    scores,
    count,
    radius,
    *,
    generator,
    power=1.5,
    exploration=0.15,
):
    """Sample an error-weighted local kernel mixture with global probe exploration."""
    if count < 1:
        raise ValueError("adaptive refinement count must be positive.")
    if radius <= 0:
        raise ValueError("adaptive refinement radius must be positive.")
    if power <= 0:
        raise ValueError("adaptive refinement power must be positive.")
    if not 0 <= exploration < 1:
        raise ValueError("adaptive refinement exploration must lie in [0, 1).")
    if candidates.ndim != 2 or scores.shape != candidates.shape[:-1]:
        raise ValueError("candidates and scores have incompatible shapes.")
    if len(candidates) < 2:
        raise ValueError("adaptive refinement requires at least two candidates.")
    if not torch.isfinite(scores).all() or torch.any(scores < 0):
        raise ValueError("adaptive refinement scores must be finite and nonnegative.")

    device, dtype = candidates.device, candidates.dtype
    dimension = candidates.shape[-1]
    local_count = max(1, round((1 - exploration) * count))
    explore_count = count - local_count
    floor = torch.finfo(dtype).eps
    scale = torch.quantile(scores, 0.99).clamp_min(floor)
    weights = ((scores.clamp(max=scale) + floor) / scale).pow(power)
    center_indices = torch.multinomial(
        weights,
        local_count,
        replacement=True,
        generator=generator,
    )
    centers = candidates[center_indices]

    pool_size = min(512, len(candidates))
    pool_indices = torch.topk(weights, pool_size, sorted=False).indices
    high_error = candidates[pool_indices]
    distances = torch.cdist(centers, high_error)
    neighbor = min(8, pool_size)
    local_radius = distances.kthvalue(neighbor, dim=-1).values
    local_radius = local_radius.clamp(min=0.005 * radius, max=0.15 * radius)
    coordinate_scale = local_radius / sqrt(dimension)

    local = centers + coordinate_scale[:, None] * torch.randn(
        local_count,
        dimension,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    invalid = torch.linalg.vector_norm(local, dim=-1) > radius
    for _ in range(20):
        if not bool(invalid.any().item()):
            break
        size = int(invalid.sum().item())
        local[invalid] = centers[invalid] + coordinate_scale[invalid, None] * torch.randn(
            size,
            dimension,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        invalid = torch.linalg.vector_norm(local, dim=-1) > radius
    local[invalid] = centers[invalid]

    if explore_count:
        if explore_count <= len(candidates):
            exploration_indices = torch.randperm(
                len(candidates),
                generator=generator,
                device=device,
            )[:explore_count]
        else:
            exploration_indices = torch.randint(
                len(candidates),
                (explore_count,),
                generator=generator,
                device=device,
            )
        return torch.cat((local, candidates[exploration_indices]), dim=0)
    return local


@torch.no_grad()
def reference_targets(reference, states, *, time_scale, batch_size):
    outputs = []
    for start in range(0, len(states), batch_size):
        values = reference(states[start : start + batch_size])
        if values.shape != states[start : start + batch_size].shape:
            raise ValueError("The internal reference evaluator changed the state shape.")
        if not torch.isfinite(values).all():
            raise FloatingPointError("The internal reference evaluator returned nonfinite values.")
        target = time_scale * values
        if not torch.isfinite(target).all():
            raise FloatingPointError("Time scaling produced nonfinite reference targets.")
        outputs.append(target.detach())
    return torch.cat(outputs, dim=0)
