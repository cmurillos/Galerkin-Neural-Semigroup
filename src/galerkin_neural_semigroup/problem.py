"""Compact public problem definition and private Galerkin supervision."""

from math import isclose
from pathlib import Path
from time import perf_counter

import torch

from ._loss import adaptive_field_score, component_weights, field_loss_terms
from ._network import _SpectralMLP
from ._sampling import (
    adaptive_refinement,
    canonical_directions,
    radial_probe,
    reference_targets,
    reference_targets_and_jacobians,
    uniform_ball,
    values_and_full_jacobian,
)
from ._validation import (
    compute_device,
    floating_dtype,
    hidden_widths,
    nonnegative_real,
    positive_integer,
    positive_real,
)
from .semigroup import NeuralSemigroup


def _basis_signature(basis):
    signature = {
        "dimension": int(basis.dimension),
        "value_shape": list(getattr(basis, "value_shape", ())),
    }
    for name in ("family", "component_sizes"):
        value = getattr(basis, name, None)
        if value is not None:
            signature[name] = list(value) if isinstance(value, tuple) else value
    return signature


def _generator(device, seed):
    target = device if device.type == "cuda" else torch.device("cpu")
    result = torch.Generator(device=target)
    result.manual_seed(seed)
    return result


def _field_values_and_jacobian(field, states, directions):
    return values_and_full_jacobian(field, states, directions)


def _mean_losses(
    field,
    states,
    targets,
    directions,
    target_jacobians,
    batch_size,
    *,
    weights,
    jacobian_weight,
):
    totals = {"loss": 0.0, "value_loss": 0.0, "jacobian_loss": 0.0}
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            target = targets[start : start + batch_size]
            prediction, prediction_jacobian = _field_values_and_jacobian(
                field,
                states[start : start + batch_size],
                directions,
            )
            value, jacobian = field_loss_terms(
                prediction,
                target,
                prediction_jacobian,
                target_jacobians[start : start + batch_size],
                weights=weights,
            )
            totals["value_loss"] += float(value.sum().item())
            totals["jacobian_loss"] += float(jacobian.sum().item())
            totals["loss"] += float((value + jacobian_weight * jacobian).sum().item())
    return {name: value / len(states) for name, value in totals.items()}


def _mean_value_loss(field, states, targets, batch_size, *, weights):
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            target = targets[start : start + batch_size]
            prediction = field(states[start : start + batch_size])
            total += float(adaptive_field_score(prediction, target, weights=weights).sum().item())
    return total / len(states)


def _refinement_scores(
    field,
    states,
    targets,
    batch_size,
    *,
    weights,
):
    scores = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            target = targets[start : start + batch_size]
            prediction = field(states[start : start + batch_size])
            scores.append(
                adaptive_field_score(
                    prediction,
                    target,
                    weights=weights,
                )
            )
    return torch.cat(scores)


def _upper_quantile(values, probability=0.99):
    return float(torch.quantile(values, probability).item())


def _epoch_order(count, recent_start, emphasize_recent, *, generator, device):
    if not emphasize_recent or recent_start >= count:
        return torch.randperm(count, generator=generator, device=device)
    recent_count = count - recent_start
    recent = recent_start + torch.randint(
        recent_count,
        (count // 2,),
        generator=generator,
        device=device,
    )
    complete = torch.randint(
        count,
        (count - len(recent),),
        generator=generator,
        device=device,
    )
    order = torch.cat((recent, complete))
    return order[torch.randperm(count, generator=generator, device=device)]


class NeuralSemigroupProblem:
    """Define and train the method without exposing the Galerkin target field.

    The normal route receives an operational ``ngfield`` basis, a complete weak
    form, and the radius of the reduced training ball. Geometry, components and
    restrictions are already carried by the basis. The initial normalized-volume
    design, adaptive refinement rule, tanh MLP, component-balanced Sobolev loss and
    exact spectral projection are fixed by the method.
    """

    def __init__(
        self,
        *,
        basis,
        weak,
        radius,
        quadrature=None,
        time_scale=1.0,
    ):
        dimension = positive_integer(getattr(basis, "dimension", None), "basis.dimension")
        if not callable(getattr(basis, "evaluate", None)):
            raise TypeError("basis must provide an evaluate method.")
        if not callable(weak):
            raise TypeError("weak must be callable.")
        self.basis = basis
        self.weak = weak
        self.radius = positive_real(radius, "radius")
        self.quadrature = quadrature
        self.time_scale = positive_real(time_scale, "time_scale")
        self.dimension = dimension
        self._explicit_problem = None

    @classmethod
    def from_galerkin(
        cls,
        problem,
        *,
        basis,
        radius,
        quadrature=None,
        time_scale=1.0,
    ):
        """Adapt an explicit legacy/general GalerkinProblem without exposing its field."""
        from ngfield import GalerkinProblem

        if not isinstance(problem, GalerkinProblem):
            raise TypeError("problem must be an ngfield.GalerkinProblem.")
        result = cls(
            basis=basis,
            weak=problem.weak,
            radius=radius,
            quadrature=quadrature,
            time_scale=time_scale,
        )
        result._explicit_problem = problem
        return result

    def _build_coordinate_system(self, *, device, dtype):
        from ngfield import GalerkinField

        options = {
            "basis": self.basis,
            "quadrature": self.quadrature,
            "device": device,
            "dtype": dtype,
        }
        if self._explicit_problem is None:
            return GalerkinField(weak=self.weak, **options)
        return self._explicit_problem.field(**options)

    def train(
        self,
        *,
        hidden,
        lipschitz,
        samples=10_000,
        batch_size=256,
        epochs=1_000,
        max_epochs=None,
        tolerance=None,
        max_time=None,
        refine_every=None,
        refine_samples=512,
        candidate_samples=32_768,
        patience=100,
        lr=1e-3,
        jacobian_weight=0.1,
        balance_epsilon=1e-6,
        seed=0,
        device="auto",
        dtype=None,
        verbose=False,
    ):
        """Train the fixed neural field, optionally refining difficult regions."""
        hidden = hidden_widths(hidden)
        lipschitz = positive_real(lipschitz, "lipschitz")
        samples = positive_integer(samples, "samples", minimum=2)
        batch_size = min(positive_integer(batch_size, "batch_size"), samples)
        epochs = positive_integer(epochs, "epochs")
        if max_epochs is None:
            max_epochs = epochs
        max_epochs = positive_integer(max_epochs, "max_epochs")
        if max_epochs < epochs:
            raise ValueError("max_epochs must be greater than or equal to epochs.")
        if tolerance is not None:
            tolerance = positive_real(tolerance, "tolerance")
        if max_time is not None:
            max_time = positive_real(max_time, "max_time")
        if refine_every is not None:
            refine_every = positive_integer(refine_every, "refine_every")
        refine_samples = positive_integer(refine_samples, "refine_samples")
        candidate_samples = positive_integer(candidate_samples, "candidate_samples", minimum=2)
        patience = positive_integer(patience, "patience")
        lr = positive_real(lr, "lr")
        jacobian_weight = nonnegative_real(jacobian_weight, "jacobian_weight")
        balance_epsilon = positive_real(balance_epsilon, "balance_epsilon")
        seed = positive_integer(seed, "seed", minimum=0)
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")
        device = compute_device(device)
        dtype = floating_dtype(dtype)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        generator = _generator(device, seed)

        coordinate_system = self._build_coordinate_system(device=device, dtype=dtype)
        if coordinate_system.dimension != self.dimension:
            raise RuntimeError("The internal coordinate field and basis dimensions differ.")

        validation_samples = max(1, min(4096, samples // 10))
        train_states = uniform_ball(
            samples,
            self.dimension,
            self.radius,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        validation_states = uniform_ball(
            validation_samples,
            self.dimension,
            self.radius,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        jacobian_basis = canonical_directions(
            self.dimension,
            self.radius,
            device=device,
            dtype=dtype,
        )
        train_targets, train_target_jacobians = reference_targets_and_jacobians(
            coordinate_system,
            train_states,
            radius=self.radius,
            time_scale=self.time_scale,
            batch_size=batch_size,
        )
        validation_targets = reference_targets(
            coordinate_system,
            validation_states,
            time_scale=self.time_scale,
            batch_size=batch_size,
        )
        jacobian_audit_samples = min(64, validation_samples)
        _, validation_target_jacobians = reference_targets_and_jacobians(
            coordinate_system,
            validation_states[:jacobian_audit_samples],
            radius=self.radius,
            time_scale=self.time_scale,
            batch_size=batch_size,
        )
        component_rms = train_targets.square().mean(dim=0).sqrt()
        weights = component_weights(train_targets, epsilon=balance_epsilon)
        candidates = None
        candidate_targets = None
        if refine_every is not None or tolerance is not None:
            candidates = radial_probe(
                candidate_samples,
                self.dimension,
                self.radius,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            candidate_targets = reference_targets(
                coordinate_system,
                candidates,
                time_scale=self.time_scale,
                batch_size=batch_size,
            )

        field = _SpectralMLP(
            self.dimension,
            hidden,
            lipschitz,
            device=device,
            dtype=dtype,
        )
        optimizer = torch.optim.Adam(field.parameters(), lr=lr)
        training_losses = []
        validation_losses = []
        training_value_losses = []
        validation_value_losses = []
        training_jacobian_losses = []
        validation_jacobian_losses = []
        validation_jacobian_audits = []
        candidate_checks = []
        refinement_events = []
        best_loss = float("inf")
        best_candidate_q99 = None
        best_epoch = 0
        best_state = None
        report_every = max(1, max_epochs // 10)
        stage_best = float("inf")
        stale_epochs = 0
        stalled_checks = 0
        recent_start = len(train_states)
        emphasize_until = 0
        completed_epochs = 0
        stop_reason = "max_epochs"
        started_at = perf_counter()

        for epoch in range(max_epochs):
            field.train()
            training_count = len(train_states)
            order = _epoch_order(
                training_count,
                recent_start,
                epoch < emphasize_until,
                generator=generator,
                device=device,
            )
            epoch_totals = {"loss": 0.0, "value_loss": 0.0, "jacobian_loss": 0.0}
            for start in range(0, training_count, batch_size):
                indices = order[start : start + batch_size]
                prediction, prediction_jacobian = _field_values_and_jacobian(
                    field,
                    train_states[indices],
                    jacobian_basis,
                )
                target = train_targets[indices]
                value, jacobian = field_loss_terms(
                    prediction,
                    target,
                    prediction_jacobian,
                    train_target_jacobians[indices],
                    weights=weights,
                )
                loss = (value + jacobian_weight * jacobian).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training produced a nonfinite loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                count = len(indices)
                epoch_totals["loss"] += float(loss.detach().item()) * count
                epoch_totals["value_loss"] += float(value.detach().sum().item())
                epoch_totals["jacobian_loss"] += float(jacobian.detach().sum().item())
            epoch_metrics = {name: value / training_count for name, value in epoch_totals.items()}
            field.eval()
            validation_value = _mean_value_loss(
                field,
                validation_states,
                validation_targets,
                batch_size,
                weights=weights,
            )
            completed_epochs = epoch + 1
            validation_jacobian = None
            audit_due = completed_epochs == 1 or completed_epochs % report_every == 0
            if audit_due:
                audit = _mean_losses(
                    field,
                    validation_states[:jacobian_audit_samples],
                    validation_targets[:jacobian_audit_samples],
                    jacobian_basis,
                    validation_target_jacobians,
                    batch_size,
                    weights=weights,
                    jacobian_weight=jacobian_weight,
                )
                validation_jacobian = audit["jacobian_loss"]
                validation_jacobian_audits.append(
                    {"epoch": completed_epochs, "jacobian_loss": validation_jacobian}
                )
            training_losses.append(epoch_metrics["loss"])
            validation_losses.append(validation_value)
            training_value_losses.append(epoch_metrics["value_loss"])
            validation_value_losses.append(validation_value)
            training_jacobian_losses.append(epoch_metrics["jacobian_loss"])
            validation_jacobian_losses.append(validation_jacobian)
            if best_candidate_q99 is None and validation_value < best_loss:
                best_loss = validation_value
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone() for name, value in field.state_dict().items()
                }
            if validation_value < stage_best * (1 - 1e-3):
                stage_best = validation_value
                stale_epochs = 0
            else:
                stale_epochs += 1
            if verbose and ((epoch + 1) % report_every == 0 or epoch == 0):
                print(
                    f"epoch={epoch + 1}/{max_epochs} "
                    f"train={epoch_metrics['loss']:.6e} "
                    f"validation={validation_value:.6e} "
                    f"samples={training_count}"
                )

            elapsed = perf_counter() - started_at
            if max_time is not None and elapsed >= max_time:
                stop_reason = "max_time"
                break

            refinement_check_due = refine_every is not None and completed_epochs % refine_every == 0
            refinement_due = refinement_check_due and stale_epochs >= patience
            accuracy_due = (
                tolerance is not None
                and completed_epochs >= epochs
                and (
                    completed_epochs == epochs or completed_epochs % (refine_every or patience) == 0
                )
            )
            if not (refinement_check_due or accuracy_due):
                continue

            candidate_scores = _refinement_scores(
                field,
                candidates,
                candidate_targets,
                batch_size,
                weights=weights,
            )
            candidate_q99 = _upper_quantile(candidate_scores)
            if best_candidate_q99 is None or candidate_q99 < best_candidate_q99:
                best_candidate_q99 = candidate_q99
                best_loss = validation_value
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone() for name, value in field.state_dict().items()
                }
            check = {
                "epoch": completed_epochs,
                "candidate_q99_loss": candidate_q99,
            }

            if max_time is not None and perf_counter() - started_at >= max_time:
                candidate_checks.append(check)
                stop_reason = "max_time"
                break

            if tolerance is not None and completed_epochs >= epochs and candidate_q99 <= tolerance:
                candidate_checks.append(check)
                stop_reason = "tolerance"
                break

            if not refinement_due:
                candidate_checks.append(check)
                continue

            training_scores = _refinement_scores(
                field,
                train_states,
                train_targets,
                batch_size,
                weights=weights,
            )
            training_q99 = _upper_quantile(training_scores)
            check["training_q99_loss"] = training_q99
            coverage_gap = candidate_q99 > 1.25 * max(
                training_q99,
                torch.finfo(dtype).eps,
            )
            check["coverage_gap"] = coverage_gap
            candidate_checks.append(check)

            needs_refinement = tolerance is None or candidate_q99 > tolerance
            if coverage_gap and needs_refinement:
                new_states = adaptive_refinement(
                    candidates,
                    candidate_scores,
                    refine_samples,
                    self.radius,
                    generator=generator,
                )
                new_targets, new_target_jacobians = reference_targets_and_jacobians(
                    coordinate_system,
                    new_states,
                    radius=self.radius,
                    time_scale=self.time_scale,
                    batch_size=batch_size,
                )
                recent_start = len(train_states)
                train_states = torch.cat((train_states, new_states), dim=0)
                train_targets = torch.cat((train_targets, new_targets), dim=0)
                train_target_jacobians = torch.cat(
                    (train_target_jacobians, new_target_jacobians), dim=0
                )
                emphasize_until = completed_epochs + max(1, min(patience, refine_every) // 2)
                refinement_events.append(
                    {
                        "epoch": completed_epochs,
                        "added_samples": len(new_states),
                        "training_samples": len(train_states),
                        "candidate_q99_loss": candidate_q99,
                        "training_q99_loss": training_q99,
                    }
                )
                best_loss = validation_value
                best_candidate_q99 = candidate_q99
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone() for name, value in field.state_dict().items()
                }
                stage_best = float("inf")
                stale_epochs = 0
                stalled_checks = 0
                if verbose:
                    print(
                        f"refinement epoch={completed_epochs} "
                        f"added={len(new_states)} "
                        f"candidate_q99={candidate_q99:.6e}"
                    )
            else:
                stalled_checks += 1
                if completed_epochs >= epochs and stalled_checks >= 2:
                    stop_reason = "stalled"
                    break

        field.load_state_dict(best_state)
        field.eval()
        elapsed_time = perf_counter() - started_at
        final_training = _mean_losses(
            field,
            train_states,
            train_targets,
            jacobian_basis,
            train_target_jacobians,
            batch_size,
            weights=weights,
            jacobian_weight=jacobian_weight,
        )
        final_validation_value = _mean_value_loss(
            field,
            validation_states,
            validation_targets,
            batch_size,
            weights=weights,
        )
        final_validation_audit = _mean_losses(
            field,
            validation_states[:jacobian_audit_samples],
            validation_targets[:jacobian_audit_samples],
            jacobian_basis,
            validation_target_jacobians,
            batch_size,
            weights=weights,
            jacobian_weight=jacobian_weight,
        )
        candidate_q99_loss = None
        if candidates is not None:
            final_candidate_scores = _refinement_scores(
                field,
                candidates,
                candidate_targets,
                batch_size,
                weights=weights,
            )
            candidate_q99_loss = _upper_quantile(final_candidate_scores)
        norms = field.spectral_norms()
        history = {
            "training_loss": training_losses,
            "validation_loss": validation_losses,
            "training_value_loss": training_value_losses,
            "validation_value_loss": validation_value_losses,
            "training_jacobian_loss": training_jacobian_losses,
            "validation_jacobian_loss": validation_jacobian_losses,
            "validation_jacobian_audits": validation_jacobian_audits,
            "candidate_checks": candidate_checks,
            "refinements": refinement_events,
        }
        metrics = {
            "best_epoch": best_epoch + 1,
            "epochs_completed": completed_epochs,
            "stop_reason": stop_reason,
            "elapsed_time": elapsed_time,
            "training_samples": len(train_states),
            "refinements": len(refinement_events),
            "candidate_q99_loss": candidate_q99_loss,
            "training_loss": final_training["loss"],
            "validation_loss": final_validation_value,
            "training_value_loss": final_training["value_loss"],
            "validation_value_loss": final_validation_value,
            "training_jacobian_loss": final_training["jacobian_loss"],
            "validation_jacobian_loss": final_validation_audit["jacobian_loss"],
            "effective_lipschitz_bound": field.effective_lipschitz_bound(),
            "layer_spectral_norms": norms,
        }
        metadata = {
            "basis": _basis_signature(self.basis),
            "reference": {
                "requested_quadrature": self.quadrature,
                "quadrature_mode": coordinate_system.quadrature_mode,
                "quadrature_order": coordinate_system.quadrature_order,
                "quadrature_tolerance": coordinate_system.quadrature_tolerance,
                "quadrature_error_estimate": coordinate_system.quadrature_error_estimate,
                "orthonormality_error": coordinate_system.orthonormality_error,
            },
            "method": {
                "measure": (
                    "adaptive-error-kernel-ball"
                    if refine_every is not None
                    else "normalized-volume-ball"
                ),
                "activation": "tanh",
                "spectral_projection": "exact",
                "loss": "component-balanced-sobolev-full-jacobian",
                "validation_objective": "component-balanced-field-values",
                "autonomous": True,
            },
            "training": {
                "samples": samples,
                "final_samples": len(train_states),
                "validation_samples": validation_samples,
                "batch_size": batch_size,
                "epochs": epochs,
                "minimum_epochs": epochs,
                "max_epochs": max_epochs,
                "epochs_completed": completed_epochs,
                "tolerance": tolerance,
                "max_time": max_time,
                "refine_every": refine_every,
                "refine_samples": refine_samples,
                "candidate_samples": candidate_samples,
                "patience": patience,
                "adaptive_power": 1.5,
                "adaptive_exploration": 0.15,
                "lr": lr,
                "jacobian_weight": jacobian_weight,
                "balance_epsilon": balance_epsilon,
                "component_rms": component_rms.tolist(),
                "jacobian_probe": "complete-radius-scaled-canonical-basis",
                "jacobian_directions": self.dimension,
                "jacobian_audit_samples": jacobian_audit_samples,
                "jacobian_audit_every": report_every,
                "seed": seed,
                "optimizer": "Adam",
                "target_mode": ("cached-and-appended" if refinement_events else "cached"),
            },
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "reference_package": "numerical-galerkin-field@9f7f71c1",
        }
        return NeuralSemigroup(
            field=field,
            coordinate_system=coordinate_system,
            radius=self.radius,
            time_scale=self.time_scale,
            history=history,
            metrics=metrics,
            metadata=metadata,
        )

    def load(self, path, *, device="auto", dtype=None):
        """Load a checkpoint against this problem's basis and weak formulation."""
        device = compute_device(device)
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
        if checkpoint.get("schema_version") != 1:
            raise ValueError("Unsupported Galerkin Neural Semigroup checkpoint schema.")
        configuration = checkpoint.get("field_configuration", {})
        if configuration.get("dimension") != self.dimension:
            raise ValueError("The checkpoint and problem basis dimensions differ.")
        if not isclose(float(checkpoint.get("radius")), self.radius, rel_tol=0, abs_tol=0):
            raise ValueError("The checkpoint and problem use different training radii.")
        if not isclose(float(checkpoint.get("time_scale")), self.time_scale, rel_tol=0, abs_tol=0):
            raise ValueError("The checkpoint and problem use different time scales.")
        if checkpoint.get("metadata", {}).get("basis") != _basis_signature(self.basis):
            raise ValueError("The checkpoint basis signature does not match this problem.")
        if dtype is None:
            saved_dtype = checkpoint.get("metadata", {}).get("dtype", "float64")
            dtype = {"float32": torch.float32, "float64": torch.float64}.get(saved_dtype)
            if dtype is None:
                raise ValueError("The checkpoint records an unsupported dtype.")
        dtype = floating_dtype(dtype)
        coordinate_system = self._build_coordinate_system(device=device, dtype=dtype)
        field = _SpectralMLP(
            self.dimension,
            configuration["hidden"],
            configuration["lipschitz"],
            device=device,
            dtype=dtype,
        )
        field.load_state_dict(checkpoint["field_state"])
        field.eval()
        return NeuralSemigroup(
            field=field,
            coordinate_system=coordinate_system,
            radius=self.radius,
            time_scale=self.time_scale,
            history=checkpoint.get("history", {}),
            metrics=checkpoint.get("metrics", {}),
            metadata=checkpoint.get("metadata", {}),
        )

    def __repr__(self):
        return (
            "NeuralSemigroupProblem("
            f"dimension={self.dimension}, radius={self.radius:g}, "
            f"time_scale={self.time_scale:g})"
        )
