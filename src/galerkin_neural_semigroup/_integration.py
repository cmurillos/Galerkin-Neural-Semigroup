"""Explicit integration of autonomous reduced neural fields."""

from math import ceil

import torch

from ._validation import positive_real

_MAX_INTERNAL_STEPS = 1_000_000


def _checked_field(field, state):
    value = field(state)
    if value.shape != state.shape:
        raise ValueError("The neural field changed the state shape during integration.")
    if not torch.isfinite(value).all():
        raise FloatingPointError("The neural field returned a nonfinite velocity.")
    return value


def _rk4_step(field, state, step):
    k1 = _checked_field(field, state)
    k2 = _checked_field(field, state + (step / 2) * k1)
    k3 = _checked_field(field, state + (step / 2) * k2)
    k4 = _checked_field(field, state + step * k3)
    result = state + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    if not torch.isfinite(result).all():
        raise FloatingPointError("RK4 produced a nonfinite state.")
    return result


def _rk45_step(field, state, step):
    k1 = _checked_field(field, state)
    k2 = _checked_field(field, state + step * (k1 / 5))
    k3 = _checked_field(field, state + step * (3 * k1 / 40 + 9 * k2 / 40))
    k4 = _checked_field(
        field,
        state + step * (44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9),
    )
    k5 = _checked_field(
        field,
        state + step * (19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729),
    )
    k6 = _checked_field(
        field,
        state
        + step
        * (
            9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656
        ),
    )
    fifth = state + step * (
        35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84
    )
    k7 = _checked_field(field, fifth)
    fourth = state + step * (
        5179 * k1 / 57600
        + 7571 * k3 / 16695
        + 393 * k4 / 640
        - 92097 * k5 / 339200
        + 187 * k6 / 2100
        + k7 / 40
    )
    if not torch.isfinite(fifth).all() or not torch.isfinite(fourth).all():
        raise FloatingPointError("RK45 produced a nonfinite state.")
    return fifth, fifth - fourth


def _fixed_interval(field, state, start, stop, maximum_step, budget):
    count = max(1, ceil(abs(stop - start) / maximum_step))
    if count > budget:
        raise RuntimeError("Time integration exceeded its internal step budget.")
    step = (stop - start) / count
    for _ in range(count):
        state = _rk4_step(field, state, step)
    return state, count


def _adaptive_interval(field, state, start, stop, tolerance, budget):
    direction = 1.0 if stop > start else -1.0
    current = start
    step_size = abs(stop - start)
    count = 0
    epsilon = torch.finfo(field.dtype).eps
    while direction * (stop - current) > 0:
        if count >= budget:
            raise RuntimeError("Time integration exceeded its internal step budget.")
        remaining = abs(stop - current)
        step_size = min(step_size, remaining)
        signed_step = direction * step_size
        candidate, difference = _rk45_step(field, state, signed_step)
        scale = tolerance * (1 + torch.maximum(torch.abs(state), torch.abs(candidate)))
        error = float(torch.max(torch.abs(difference) / scale).detach().item())
        count += 1
        minimum = 16 * epsilon * max(1.0, abs(current), abs(stop))
        if error <= 1:
            state = candidate
            current += signed_step
            if abs(stop - current) <= minimum:
                current = stop
            factor = 5.0 if error == 0 else min(5.0, max(0.2, 0.9 * error ** (-0.2)))
            step_size *= factor
        else:
            if step_size <= minimum:
                raise RuntimeError(
                    "RK45 cannot satisfy the requested tolerance at machine precision."
                )
            step_size *= max(0.2, 0.9 * error ** (-0.2))
    return state, count


def solve(field, state, times, *, step=None, tolerance=None):
    field._states(state)
    if not torch.isfinite(state).all():
        raise ValueError("The initial state must be finite.")
    if not isinstance(times, torch.Tensor):
        raise TypeError("times must be a torch tensor.")
    if times.ndim != 1 or len(times) < 1:
        raise ValueError("times must have shape [T] with T >= 1.")
    if times.device != field.device or times.dtype != field.dtype:
        raise ValueError("times and the neural field must share device and dtype.")
    if not torch.isfinite(times).all():
        raise ValueError("times must be finite.")
    if len(times) > 1:
        increments = times[1:] - times[:-1]
        if not bool(torch.all(increments > 0).item()) and not bool(
            torch.all(increments < 0).item()
        ):
            raise ValueError("times must be strictly monotone.")
    if step is not None and tolerance is not None:
        raise ValueError("step and tolerance select different solvers and cannot be combined.")
    fixed_step = None if step is None else positive_real(step, "step")
    if tolerance is None:
        adaptive_tolerance = 5e-5 if field.dtype == torch.float32 else 1e-8
    else:
        adaptive_tolerance = positive_real(tolerance, "tolerance")
        if adaptive_tolerance >= 1:
            raise ValueError("tolerance must be strictly smaller than one.")
    if not state.numel() or len(times) == 1:
        return torch.stack([state] * len(times), dim=0)
    host_times = times.detach().cpu().tolist()
    states = [state]
    used_steps = 0
    for start, stop in zip(host_times[:-1], host_times[1:]):
        budget = _MAX_INTERNAL_STEPS - used_steps
        if fixed_step is None:
            state, count = _adaptive_interval(field, state, start, stop, adaptive_tolerance, budget)
        else:
            state, count = _fixed_interval(field, state, start, stop, fixed_step, budget)
        used_steps += count
        states.append(state)
    return torch.stack(states, dim=0)
