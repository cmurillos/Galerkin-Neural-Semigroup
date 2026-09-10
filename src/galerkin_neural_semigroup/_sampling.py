"""Sampling and target preparation fixed by the learning functional."""

from math import sqrt

import torch


def _unit_directions(count, dimension, *, generator, device, dtype):
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
    return directions / norms


def initial_radial_layers(count, dimension, radius, *, device, dtype):
    """Return the fixed initial radii, including the origin and boundary."""
    if count < 2:
        raise ValueError("radial sampling requires at least two states.")
    if dimension < 1:
        raise ValueError("radial sampling dimension must be positive.")
    if radius <= 0:
        raise ValueError("radial sampling radius must be positive.")
    layer_count = min(count, max(2, min(32, round(sqrt(count / dimension)))))
    return torch.linspace(0.0, radius, layer_count, device=device, dtype=dtype)


def radial_design(count, dimension, layers, *, generator, device, dtype):
    """Sample unit-sphere directions on prescribed concentric radial layers."""
    if count < 1:
        raise ValueError("radial design count must be positive.")
    if dimension < 1:
        raise ValueError("radial design dimension must be positive.")
    if layers.ndim != 1 or len(layers) < 1:
        raise ValueError("radial layers must be a nonempty vector.")
    if layers.device != device or layers.dtype != dtype:
        raise ValueError("radial layers must use the requested device and dtype.")
    if not torch.isfinite(layers).all() or torch.any(layers < 0):
        raise ValueError("radial layers must be finite and nonnegative.")
    if len(layers) > 1 and torch.any(layers[1:] <= layers[:-1]):
        raise ValueError("radial layers must be strictly increasing.")
    if count < len(layers):
        raise ValueError("radial design count must cover every layer.")

    if len(layers) == 1:
        layer_indices = torch.zeros(count, device=device, dtype=torch.long)
    else:
        positive = torch.arange(1, len(layers), device=device)
        repeats = (count - 1 + len(positive) - 1) // len(positive)
        positive = positive.repeat(repeats)[: count - 1]
        positive = positive[torch.randperm(len(positive), generator=generator, device=device)]
        layer_indices = torch.cat(
            (torch.zeros(1, device=device, dtype=torch.long), positive),
            dim=0,
        )
    directions = _unit_directions(
        count,
        dimension,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    states = layers[layer_indices, None] * directions
    states[layer_indices == 0] = 0
    return states, layer_indices


def radial_interval_probe(count, dimension, layers, *, generator, device, dtype):
    """Sample unit-sphere directions at every adjacent-layer midpoint."""
    if dimension < 1:
        raise ValueError("radial probe dimension must be positive.")
    if layers.ndim != 1 or len(layers) < 2:
        raise ValueError("radial probe requires at least two layers.")
    if layers.device != device or layers.dtype != dtype:
        raise ValueError("radial layers must use the requested device and dtype.")
    if not torch.isfinite(layers).all() or torch.any(layers < 0):
        raise ValueError("radial layers must be finite and nonnegative.")
    if torch.any(layers[1:] <= layers[:-1]):
        raise ValueError("radial layers must be strictly increasing.")

    interval_count = len(layers) - 1
    if count < interval_count:
        raise ValueError("radial probe count must cover every interval.")
    interval_indices = torch.arange(interval_count, device=device)
    repeats = (count + interval_count - 1) // interval_count
    interval_indices = interval_indices.repeat(repeats)[:count]
    interval_indices = interval_indices[
        torch.randperm(len(interval_indices), generator=generator, device=device)
    ]
    midpoints = 0.5 * (layers[:-1] + layers[1:])
    directions = _unit_directions(
        count,
        dimension,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return midpoints[interval_indices, None] * directions, interval_indices


def canonical_directions(dimension, radius, *, device, dtype):
    """Return the complete radius-scaled canonical basis."""
    return radius * torch.eye(dimension, device=device, dtype=dtype)


def radial_error_profile(scores, radial_indices, radii):
    """Average the independent value loss over directions at each probe radius."""
    if scores.ndim != 1 or radial_indices.shape != scores.shape:
        raise ValueError("scores and radial indices must be matching vectors.")
    if radial_indices.dtype != torch.long:
        raise ValueError("radial indices must be integers.")
    if radii.ndim != 1 or len(radii) < 1:
        raise ValueError("probe radii must be a nonempty vector.")
    if torch.any(radial_indices < 0) or torch.any(radial_indices >= len(radii)):
        raise ValueError("radial indices fall outside the supplied probe radii.")
    if not torch.isfinite(scores).all() or torch.any(scores < 0):
        raise ValueError("radial scores must be finite and nonnegative.")

    radial_errors = torch.empty(len(radii), device=scores.device, dtype=scores.dtype)
    for index in range(len(radii)):
        shell_scores = scores[radial_indices == index]
        if len(shell_scores) == 0:
            raise ValueError("every radial layer must have candidate scores.")
        radial_errors[index] = shell_scores.mean()
    return radial_errors


def adaptive_refinement(
    candidates,
    scores,
    interval_indices,
    layers,
    count,
    *,
    generator,
):
    """Insert a spherical layer inside the worst adjacent radial interval."""
    if count < 1:
        raise ValueError("adaptive refinement count must be positive.")
    if candidates.ndim != 2 or scores.shape != candidates.shape[:-1]:
        raise ValueError("candidates and scores have incompatible shapes.")
    if layers.ndim != 1 or len(layers) < 2:
        raise ValueError("adaptive refinement requires at least two radial layers.")

    device, dtype = candidates.device, candidates.dtype
    dimension = candidates.shape[-1]
    midpoints = 0.5 * (layers[:-1] + layers[1:])
    radial_errors = radial_error_profile(scores, interval_indices, midpoints)

    interval_index = int(torch.argmax(radial_errors).item())
    new_radius = midpoints[interval_index]
    directions = _unit_directions(
        count,
        dimension,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    states = new_radius * directions
    return states, new_radius, radial_errors, interval_index


def values_and_full_jacobian(function, states, directions):
    """Evaluate a batched map and all directional Jacobian columns."""
    if states.ndim != 2:
        raise ValueError("states must have shape [samples,dimension].")
    if directions.shape != (states.shape[-1], states.shape[-1]):
        raise ValueError("directions must be a square basis for the state dimension.")
    values = None
    columns = []
    for direction in directions:
        current, derivative = torch.func.jvp(
            function,
            (states,),
            (direction.expand_as(states),),
        )
        if values is None:
            values = current
        columns.append(derivative)
    jacobian = torch.stack(columns, dim=-1)
    if values.shape != states.shape or jacobian.shape != (*states.shape, states.shape[-1]):
        raise ValueError("The evaluated map changed the state or Jacobian shape.")
    return values, jacobian


def reference_targets_and_jacobians(
    reference,
    states,
    *,
    radius,
    time_scale,
    batch_size,
):
    """Return detached field values and complete Jacobians in batches."""
    directions = canonical_directions(
        states.shape[-1],
        radius,
        device=states.device,
        dtype=states.dtype,
    )
    outputs = []
    derivatives = []
    for start in range(0, len(states), batch_size):
        state = states[start : start + batch_size]
        values, jacobian = values_and_full_jacobian(reference, state, directions)
        target = time_scale * values
        target_jacobian = time_scale * jacobian
        if not torch.isfinite(target).all() or not torch.isfinite(target_jacobian).all():
            raise FloatingPointError("The internal reference evaluator returned nonfinite data.")
        outputs.append(target.detach())
        derivatives.append(target_jacobian.detach())
    return torch.cat(outputs, dim=0), torch.cat(derivatives, dim=0)


def reference_targets(reference, states, *, time_scale, batch_size):
    """Return detached field values in batches without derivative materialization."""
    outputs = []
    for start in range(0, len(states), batch_size):
        target = time_scale * reference(states[start : start + batch_size])
        if target.shape != states[start : start + batch_size].shape:
            raise ValueError("The internal reference evaluator changed the state shape.")
        if not torch.isfinite(target).all():
            raise FloatingPointError("The internal reference evaluator returned nonfinite data.")
        outputs.append(target.detach())
    return torch.cat(outputs, dim=0)
