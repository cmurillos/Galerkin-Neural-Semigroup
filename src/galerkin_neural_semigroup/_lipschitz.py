"""Reproducible empirical local derivative estimates of the Galerkin field."""

from math import isfinite

import torch

from ._sampling import sample_unit_ball, unit_ball_targets


def estimate_reference_lipschitz(
    reference, dimension, *, radius, batch_size, generator, device, dtype
):
    """Estimate the largest sampled ``||Dg_R(x)||_2``; this is not a certificate."""
    probes = max(16, min(64, 1024 // dimension))
    anchors = sample_unit_ball(
        probes - 1, dimension, generator=generator, device=device, dtype=dtype
    )
    anchors = torch.cat((torch.zeros(1, dimension, device=device, dtype=dtype), anchors))
    targets = unit_ball_targets(reference, anchors, radius=radius, order=1, batch_size=batch_size)
    axes = tuple(tuple(int(i == j) for i in range(dimension)) for j in range(dimension))
    derivative_columns = torch.stack([targets[alpha] for alpha in axes], dim=-1)
    result = float(torch.linalg.matrix_norm(derivative_columns, ord=2).max().item())
    if not isfinite(result):
        raise FloatingPointError("Reference derivative estimation produced a nonfinite value.")
    return result, probes
