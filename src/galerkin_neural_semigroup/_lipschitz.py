"""Reproducible local finite-difference estimates of the private reference field."""

from math import isfinite

import torch

from ._sampling import sample_unit_ball, unit_ball_targets


@torch.no_grad()
def estimate_reference_lipschitz(
    reference, dimension, *, radius, time_scale, batch_size, generator, device, dtype
):
    """Estimate Lip((tau/R) G(R*x)) on the unit ball; never certify a bound."""
    # Each difference quotient uses two states strictly inside the training ball.
    # Including the origin helps detect the linear part of affine fields.
    probes = max(16, min(64, 1024 // dimension))
    anchors = sample_unit_ball(
        probes - 1,
        dimension,
        sampling="mixed",
        generator=generator,
        device=device,
        dtype=dtype,
    )
    anchors = torch.cat((torch.zeros(1, dimension, device=device, dtype=dtype), 0.98 * anchors))
    step = 1e-2 if dtype == torch.float32 else 1e-3
    axes = step * torch.eye(dimension, device=device, dtype=dtype)
    plus = (anchors[:, None, :] + axes[None, :, :]).reshape(-1, dimension)
    minus = (anchors[:, None, :] - axes[None, :, :]).reshape(-1, dimension)
    plus_values = unit_ball_targets(
        reference, plus, radius=radius, time_scale=time_scale, batch_size=batch_size
    ).reshape(probes, dimension, dimension)
    minus_values = unit_ball_targets(
        reference, minus, radius=radius, time_scale=time_scale, batch_size=batch_size
    ).reshape(probes, dimension, dimension)
    # Column j is the measured difference quotient in coordinate direction j.
    jacobians = ((plus_values - minus_values) / (2 * step)).transpose(-1, -2)
    estimates = torch.linalg.matrix_norm(jacobians, ord=2)
    result = float(estimates.max().item())
    if not isfinite(result):
        raise FloatingPointError("Reference Lipschitz estimation produced a nonfinite value.")
    return result, probes, step
