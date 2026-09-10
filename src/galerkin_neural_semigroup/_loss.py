"""Component-balanced Sobolev supervision for reduced vector fields."""

import torch


def component_weights(targets, *, epsilon):
    """Return fixed inverse-RMS weights with a relative component floor."""
    if targets.ndim != 2:
        raise ValueError("targets must have shape [samples,dimension].")
    component_energy = targets.square().mean(dim=0)
    mean_energy = component_energy.mean()
    if float(mean_energy.item()) == 0.0:
        return torch.ones_like(component_energy)
    return torch.rsqrt(component_energy + epsilon * mean_energy)


def field_loss_terms(prediction, target, prediction_jvp, target_jvp, *, weights):
    """Return per-state component-balanced value and Jacobian errors."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if prediction_jvp.shape != target_jvp.shape or prediction_jvp.shape != target.shape:
        raise ValueError("Jacobian-vector products and targets must have the same shape.")
    if weights.shape != target.shape[-1:]:
        raise ValueError("weights must contain one value per field component.")
    value = ((prediction - target) * weights).square().mean(dim=-1)
    jacobian = ((prediction_jvp - target_jvp) * weights).square().mean(dim=-1)
    return value, jacobian


def field_loss(
    prediction,
    target,
    prediction_jvp,
    target_jvp,
    *,
    weights,
    jacobian_weight,
):
    """Return the balanced value loss plus weighted JVP loss."""
    value, jacobian = field_loss_terms(
        prediction,
        target,
        prediction_jvp,
        target_jvp,
        weights=weights,
    )
    return (value + jacobian_weight * jacobian).mean()


def adaptive_field_score(
    prediction,
    target,
    prediction_jvp,
    target_jvp,
    *,
    weights,
    jacobian_weight,
):
    """Return the per-state balanced Sobolev score used by refinement."""
    value, jacobian = field_loss_terms(
        prediction,
        target,
        prediction_jvp,
        target_jvp,
        weights=weights,
    )
    return value + jacobian_weight * jacobian
