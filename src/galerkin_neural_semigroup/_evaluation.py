"""Independent diagnostics of a learned field against its private NGF reference."""

import torch
from ngfield import DomainExitError, integrate_field, multi_indices, state_derivatives, time_error

from ._network import OPEN_BALL_DOMAIN
from ._sampling import unit_ball_targets
from ._validation import positive_integer


def _states(model, states, name):
    if model.field.configuration().get("domain") != OPEN_BALL_DOMAIN:
        raise ValueError("Independent evaluation requires a new open-ball model.")
    if not isinstance(states, torch.Tensor) or states.ndim != 2 or not len(states):
        raise ValueError(f"{name} must be a nonempty tensor with shape [samples,N].")
    model.field._states(states)
    return states / model.radius


def _radial_edges(edges, states):
    if edges is None:
        return torch.linspace(0, 1, 6, dtype=states.dtype, device=states.device)
    result = torch.as_tensor(edges, dtype=states.dtype, device=states.device)
    if (
        result.ndim != 1
        or len(result) < 2
        or not bool(torch.isfinite(result).all())
        or not bool(torch.all(result[1:] > result[:-1]))
        or result[0] != 0
        or result[-1] != 1
    ):
        raise ValueError("radial_edges must increase strictly from 0 to 1.")
    return result


def evaluate_field(model, states, *, radial_edges=None, batch_size=128):
    """Report normalized field, H^k and sampled-Lipschitz errors on reserved states."""
    x = _states(model, states, "states")
    edges = _radial_edges(radial_edges, x)
    batch_size = positive_integer(batch_size, "batch_size")
    order = model.field.sobolev_order
    indices = multi_indices(model.dimension, max(1, order))
    axes = tuple(tuple(int(i == j) for i in range(model.dimension)) for j in range(model.dimension))
    squared = torch.zeros(order + 1, device=model.device, dtype=torch.float64)
    radial_squared = torch.zeros(len(edges) - 1, device=model.device, dtype=torch.float64)
    radial_counts = torch.zeros(len(edges) - 1, device=model.device, dtype=torch.long)
    lipschitz_g = x.new_zeros(())
    lipschitz_f = x.new_zeros(())
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size]
            target = unit_ball_targets(
                model._coordinate_system,
                batch,
                radius=model.radius,
                order=max(1, order),
                batch_size=batch_size,
            )
            predicted = state_derivatives(model.field.core, batch, max(1, order))
            value_error = None
            for alpha in indices:
                if sum(alpha) > order:
                    continue
                error = (predicted[alpha] - target[alpha]).square().sum(dim=-1)
                squared[sum(alpha)] += error.double().sum()
                if not any(alpha):
                    value_error = error
            bins = torch.bucketize(torch.linalg.vector_norm(batch, dim=-1), edges[1:-1])
            radial_squared.index_add_(0, bins, value_error.double())
            radial_counts += torch.bincount(bins, minlength=len(radial_counts))
            lipschitz_g = torch.maximum(
                lipschitz_g,
                torch.linalg.matrix_norm(torch.stack([target[a] for a in axes], dim=-1), ord=2)
                .max()
                .detach(),
            )
            lipschitz_f = torch.maximum(
                lipschitz_f,
                torch.linalg.matrix_norm(torch.stack([predicted[a] for a in axes], dim=-1), ord=2)
                .max()
                .detach(),
            )
    by_order = torch.sqrt((squared / len(x)).to(x.dtype))
    radial = torch.sqrt((radial_squared / radial_counts.clamp_min(1)).to(x.dtype))
    radial = radial.masked_fill(radial_counts == 0, float("nan"))
    return {
        "field_rmse": by_order[0],
        "derivative_rmse_by_order": by_order,
        "sobolev_rmse": torch.sqrt(by_order.square().sum()),
        "radial_edges": edges,
        "radial_counts": radial_counts,
        "radial_rmse": radial,
        "reference_lipschitz_sample": lipschitz_g,
        "learned_lipschitz_sample": lipschitz_f,
        "learned_lipschitz_bound": x.new_tensor(model.field.effective_lipschitz_bound()),
    }


def _population(field, states, times, *, radius, order, step, tolerance):
    """Integrate batches; split only groups containing a path that exits."""
    paths = states.new_full((len(times), *states.shape), float("nan"))
    exit_times = states.new_full((len(states),), float("nan"))

    def fill(indices):
        try:
            solved = integrate_field(
                field,
                states[indices],
                times,
                radius=radius,
                order=order,
                step=step,
                tolerance=tolerance,
            )
        except DomainExitError as exc:
            if len(indices) > 1:
                middle = len(indices) // 2
                fill(indices[:middle])
                fill(indices[middle:])
                return
            index = int(indices[0])
            exit_times[index] = exc.time
            length = int((times <= exc.time).sum().item())
            # A prefix ends at an output time already accepted by the solver.
            if length:
                paths[:length, index] = integrate_field(
                    field,
                    states[index],
                    times[:length],
                    radius=radius,
                    order=order,
                    step=step,
                    tolerance=tolerance,
                )
        else:
            paths[:, indices] = solved.detach()

    with torch.no_grad():
        fill(torch.arange(len(states), device=states.device))
    return paths, exit_times


def _mean(values, mask):
    count = mask.sum(dim=1)
    expanded = mask.reshape((*mask.shape, *((1,) * (values.ndim - 2))))
    total = torch.where(expanded, values, 0).sum(dim=1)
    divisor = count.clamp_min(1).reshape((len(count), *((1,) * (values.ndim - 2))))
    return (total / divisor).masked_fill(
        (count == 0).reshape((len(count), *((1,) * (values.ndim - 2)))), float("nan")
    )


def _velocities(model, paths, *, reference, batch_size=128):
    """Evaluate a field only at finite interior states, in bounded batches."""
    valid = torch.isfinite(paths).all(dim=-1)
    output = torch.full_like(paths, float("nan"))
    states = paths[valid]
    field = model._coordinate_system if reference else model.field
    with torch.no_grad():
        chunks = []
        for start in range(0, len(states), batch_size):
            batch = states[start : start + batch_size]
            chunks.append(field(batch).detach())
        if chunks:
            output[valid] = torch.cat(chunks)
    return output


def _refinement(field, states, times, *, radius, order, step, tolerance):
    """Reuse NGF's indicator and leave undefined values after numerical exit."""
    result = states.new_full((len(times), len(states)), float("nan"))

    def fill(indices):
        try:
            values = time_error(
                field,
                states[indices],
                times,
                radius=radius,
                order=order,
                step=step,
                tolerance=tolerance,
            )
        except DomainExitError as exc:
            if len(indices) > 1:
                middle = len(indices) // 2
                fill(indices[:middle])
                fill(indices[middle:])
                return
            index = int(indices[0])
            length = int((times <= exc.time).sum().item())
            while length:
                try:
                    result[:length, index] = time_error(
                        field,
                        states[index],
                        times[:length],
                        radius=radius,
                        order=order,
                        step=step,
                        tolerance=tolerance,
                    )
                except DomainExitError:
                    length -= 1
                else:
                    break
        else:
            result[:, indices] = values.detach()

    with torch.no_grad():
        fill(torch.arange(len(states), device=states.device))
    return result


def evaluate_trajectories(
    model, initial_states, times, *, order=4, step=None, tolerance=None, refine=False
):
    """Compare local flows and observables on reserved initial states and times."""
    _states(model, initial_states, "initial_states")
    if (
        not isinstance(times, torch.Tensor)
        or times.ndim != 1
        or len(times) < 2
        or times.device != model.device
        or times.dtype != model.dtype
        or not bool(torch.isfinite(times).all())
        or not bool(torch.all(times[1:] > times[:-1]))
    ):
        raise ValueError("times must be a finite, increasing tensor [T] on the model device.")
    if not isinstance(refine, bool):
        raise TypeError("refine must be a boolean.")
    reference = model._coordinate_system
    common = dict(radius=model.radius, order=order, step=step, tolerance=tolerance)
    g_paths, g_exit = _population(reference, initial_states, times, **common)
    f_paths, f_exit = _population(model.field, initial_states, times, **common)
    g_path, f_path = g_paths / model.radius, f_paths / model.radius
    g_alive = torch.isfinite(g_path).all(dim=-1)
    f_alive = torch.isfinite(f_path).all(dim=-1)
    both = g_alive & f_alive
    error = torch.linalg.vector_norm(f_path - g_path, dim=-1)
    mean_error = _mean(error, both)
    variance = _mean((error - mean_error[:, None]).square(), both)
    count = both.sum(dim=1)
    std_error = torch.sqrt(variance * count / (count - 1).clamp_min(1)).masked_fill(
        count < 2, float("nan")
    )

    # Physical states are used for the prepared fields; all reported rates use x=z/R.
    g_at_g = _velocities(model, g_paths, reference=True) / model.radius
    g_at_f = _velocities(model, f_paths, reference=True) / model.radius
    f_at_f = _velocities(model, f_paths, reference=False) / model.radius
    same_state_gap = f_at_f - g_at_f
    field_on_path = torch.linalg.vector_norm(same_state_gap, dim=-1)
    integral_weights = reference.integral_weights()
    weights = integral_weights.reshape(model.dimension, -1)
    shape = integral_weights.shape[1:]
    g_mass = (g_path @ weights).reshape((*g_alive.shape, *shape))
    f_mass = (f_path @ weights).reshape((*f_alive.shape, *shape))
    initial_mass = (initial_states / model.radius @ weights).reshape((len(initial_states), *shape))
    mass_gap = (f_mass - g_mass).abs()
    mass_rate_gap = (same_state_gap @ weights).reshape((*f_alive.shape, *shape)).abs()
    g_norm = torch.linalg.vector_norm(g_path, dim=-1)
    f_norm = torch.linalg.vector_norm(f_path, dim=-1)
    initial_norm = torch.linalg.vector_norm(initial_states / model.radius, dim=-1)
    radial_gap = (f_path * same_state_gap).sum(dim=-1).abs()

    report = {
        "times": times.detach().clone(),
        "reference_states": g_path,
        "learned_states": f_path,
        "reference_exit_time": g_exit,
        "learned_exit_time": f_exit,
        "reference_survival": g_alive.to(model.dtype).mean(dim=1),
        "learned_survival": f_alive.to(model.dtype).mean(dim=1),
        "common_count": count,
        "trajectory_error": error,
        "trajectory_error_mean": mean_error,
        "trajectory_error_std": std_error,
        "visited_field_rmse": torch.sqrt(_mean(field_on_path.square(), f_alive)),
        "mass_gap_mean": _mean(mass_gap, both),
        "mass_change_reference": _mean(g_mass - initial_mass, g_alive),
        "mass_change_learned": _mean(f_mass - initial_mass, f_alive),
        "mass_rate_gap_mean": _mean(mass_rate_gap, f_alive),
        "norm_gap_mean": _mean((f_norm - g_norm).abs(), both),
        "norm_change_reference": _mean(g_norm - initial_norm, g_alive),
        "norm_change_learned": _mean(f_norm - initial_norm, f_alive),
        "radial_rate_reference": _mean((g_path * g_at_g).sum(dim=-1), g_alive),
        "radial_rate_learned": _mean((f_path * f_at_f).sum(dim=-1), f_alive),
        "radial_rate_gap_mean": _mean(radial_gap, f_alive),
    }
    if refine:
        report["reference_refinement"] = (
            _refinement(reference, initial_states, times, **common) / model.radius
        )
        report["learned_refinement"] = (
            _refinement(model.field, initial_states, times, **common) / model.radius
        )
        g_valid = torch.isfinite(report["reference_refinement"])
        f_valid = torch.isfinite(report["learned_refinement"])
        report["reference_refinement_mean"] = _mean(report["reference_refinement"], g_valid)
        report["learned_refinement_mean"] = _mean(report["learned_refinement"], f_valid)
        report["reference_refinement_count"] = g_valid.sum(dim=1)
        report["learned_refinement_count"] = f_valid.sum(dim=1)
    return report
