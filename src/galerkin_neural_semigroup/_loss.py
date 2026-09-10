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


def field_value_error(prediction, target, *, weights):
    """Return the per-state component-balanced value error."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if weights.shape != target.shape[-1:]:
        raise ValueError("weights must contain one value per field component.")
    return ((prediction - target) * weights).square().mean(dim=-1)


def field_loss_terms(
    prediction,
    target,
    prediction_jacobian,
    target_jacobian,
    *,
    weights,
):
    """Return per-state component-balanced value and full-Jacobian errors."""
    value = field_value_error(prediction, target, weights=weights)
    expected = (*target.shape, target.shape[-1])
    if prediction_jacobian.shape != target_jacobian.shape:
        raise ValueError("prediction and target Jacobians must have the same shape.")
    if prediction_jacobian.shape != expected:
        raise ValueError("Jacobians must have shape [samples,dimension,dimension].")
    scaled = (prediction_jacobian - target_jacobian) * weights[None, :, None]
    jacobian = scaled.square().mean(dim=(-2, -1))
    return value, jacobian


def field_loss(
    prediction,
    target,
    prediction_jacobian,
    target_jacobian,
    *,
    weights,
    jacobian_weight,
):
    """Return the balanced value loss plus weighted full-Jacobian loss."""
    value, jacobian = field_loss_terms(
        prediction,
        target,
        prediction_jacobian,
        target_jacobian,
        weights=weights,
    )
    return (value + jacobian_weight * jacobian).mean()


def adaptive_field_score(prediction, target, *, weights):
    """Return the independent value score used by adaptive refinement."""
    return field_value_error(prediction, target, weights=weights)
