"""Relative and angular supervision for reduced vector fields."""

import torch


def field_loss_terms(prediction, target, *, epsilon):
    """Return the per-state relative and stabilized angular errors."""
    residual_squared = (prediction - target).square().sum(dim=-1)
    prediction_squared = prediction.square().sum(dim=-1)
    target_squared = target.square().sum(dim=-1)
    relative = residual_squared / (target_squared + epsilon)
    cosine = (prediction * target).sum(dim=-1) / torch.sqrt(
        (prediction_squared + epsilon) * (target_squared + epsilon)
    )
    angular = 1 - cosine.clamp(-1, 1)
    return relative, angular


def field_loss(prediction, target, *, angular_weight, epsilon):
    """Return the batch mean of relative error plus weighted angular error."""
    relative, angular = field_loss_terms(prediction, target, epsilon=epsilon)
    return (relative + angular_weight * angular).mean()
