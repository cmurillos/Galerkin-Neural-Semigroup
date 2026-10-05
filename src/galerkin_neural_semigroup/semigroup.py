"""A trained autonomous neural field and its induced flow."""

from math import isfinite
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import torch
from ngfield import integrate_field

from ._network import (
    COMPACT_SUPPORT,
    LEGACY_COMPACT_SUPPORT,
    OPEN_BALL_DOMAIN,
    UNIT_BALL_NORMALIZATION,
)
from ._validation import positive_real


class NeuralSemigroup:
    """The learned reduced flow, with optional physical-coordinate operations.

    Instances are returned by :meth:`NeuralSemigroupProblem.train` or
    :meth:`NeuralSemigroupProblem.load`; users do not construct them directly.
    ``field`` is the learned autonomous field in physical reduced coordinates.
    Unit-ball normalization and the Galerkin reference used during training remain
    internal.
    """

    def __init__(
        self,
        *,
        field,
        coordinate_system,
        radius,
        time_scale=None,
        history=None,
        metrics=None,
        metadata=None,
    ):
        self.field = field.eval()
        self._coordinate_system = coordinate_system
        self.radius = float(radius)
        self._legacy_time_scale = (
            1.0 if time_scale is None else positive_real(time_scale, "time_scale")
        )
        self.history = MappingProxyType(
            {name: tuple(values) for name, values in dict(history or {}).items()}
        )
        self.metrics = MappingProxyType(dict(metrics or {}))
        self.metadata = MappingProxyType(dict(metadata or {}))

    @property
    def dimension(self):
        return self.field.dimension

    @property
    def device(self):
        return self.field.device

    @property
    def dtype(self):
        return self.field.dtype

    @property
    def basis(self):
        return self._coordinate_system.basis

    @property
    def space(self):
        return self._coordinate_system.space

    @property
    def geometry(self):
        return self._coordinate_system.geometry

    def velocity(self, states):
        """Evaluate the learned velocity in physical time coordinates."""
        return self.field(states) / self._legacy_time_scale

    def velocity_zero_extended(self, states):
        """Evaluate the exterior-zero convention separately from the local ODE."""
        if self.field.configuration().get("exterior_convention") != "zero":
            raise ValueError("The exterior-zero convention requires a schema-6 model.")
        return self.field.zero_extended(states)

    def solve(self, z0, times, *, step=None, tolerance=None, order=4):
        """Evolve reduced coordinates from the first requested physical time.

        Uses the same indexed Taylor-jet integrator as Numerical Galerkin Field.
        """
        if not isinstance(z0, torch.Tensor):
            z0 = torch.as_tensor(z0, device=self.device, dtype=self.dtype)
        if not isinstance(times, torch.Tensor):
            times = torch.as_tensor(times, device=self.device, dtype=self.dtype)
        if self.field.configuration().get("domain") == OPEN_BALL_DOMAIN:
            integration_times, integration_step, radius = times, step, self.radius
        else:
            integration_times = times / self._legacy_time_scale
            integration_step = (
                None if step is None else positive_real(step, "step") / self._legacy_time_scale
            )
            radius = None
        return integrate_field(
            self.field,
            z0,
            integration_times,
            step=integration_step,
            tolerance=tolerance,
            order=order,
            radius=radius,
        )

    def __call__(self, state, time, *, step=None, tolerance=None, order=4):
        """Apply the learned flow at one physical time measured from zero."""
        if isinstance(time, bool) or not isinstance(time, Real):
            raise TypeError("time must be a finite real number.")
        time = float(time)
        if not isfinite(time):
            raise ValueError("time must be finite.")
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, device=self.device, dtype=self.dtype)
        if time == 0:
            self.field._states(state)
            return state.clone()
        times = torch.tensor([0.0, time], device=self.device, dtype=self.dtype)
        return self.solve(state, times, step=step, tolerance=tolerance, order=order)[-1]

    def project(self, source, *, quadrature=None):
        """Project a physical state using the fixed operational basis."""
        return self._coordinate_system.project(source, quadrature=quadrature)

    def reconstruct(self, states, points=None, *, cells=None, boundary=None):
        """Reconstruct reduced states in physical coordinates."""
        return self._coordinate_system.reconstruct(
            states,
            points,
            cells=cells,
            boundary=boundary,
        )

    def solve_physical(
        self,
        initial_state,
        times,
        points,
        *,
        projection_quadrature=None,
        cells=None,
        boundary=None,
        step=None,
        tolerance=None,
        order=4,
    ):
        """Project, evolve and reconstruct a physical initial state."""
        z0 = self.project(initial_state, quadrature=projection_quadrature)
        states = self.solve(z0, times, step=step, tolerance=tolerance, order=order)
        return self.reconstruct(states, points, cells=cells, boundary=boundary)

    def defect(self, state, first_time, second_time, *, step=None, tolerance=None, order=4):
        """Return Psi_t(Psi_s(z)) - Psi_{t+s}(z) for integrator diagnostics."""
        after_first = self(state, first_time, step=step, tolerance=tolerance, order=order)
        composed = self(after_first, second_time, step=step, tolerance=tolerance, order=order)
        direct = self(
            state,
            float(first_time) + float(second_time),
            step=step,
            tolerance=tolerance,
            order=order,
        )
        return composed - direct

    def evaluate_field(self, states, *, radial_edges=None, batch_size=128):
        """Evaluate normalized errors on explicitly reserved interior states.

        ``states`` has physical reduced coordinates ``[samples,N]``. The
        returned metrics refer to the normalized unit ball and do not enter
        training or the saved checkpoint.
        """
        from ._evaluation import evaluate_field

        return evaluate_field(self, states, radial_edges=radial_edges, batch_size=batch_size)

    def evaluate_trajectories(
        self, initial_states, times, *, step=None, tolerance=None, order=4, refine=False
    ):
        """Compare normalized local flows on reserved data in physical time.

        Initial states have physical reduced coordinates ``[samples,N]``;
        ``times`` is a strictly monotone tensor, forwards or backwards in time. Exit times and missing
        values report numerical domain exit, not certified exact exit.
        """
        from ._evaluation import evaluate_trajectories

        return evaluate_trajectories(
            self,
            initial_states,
            times,
            step=step,
            tolerance=tolerance,
            order=order,
            refine=refine,
        )

    def save(self, path):
        """Save network parameters and reproducibility metadata, but not the oracle."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        configuration = self.field.configuration()
        if configuration.get("weight_constraint") == "none":
            schema_version = 6
        elif configuration.get("domain") == OPEN_BALL_DOMAIN:
            schema_version = 5
        elif configuration.get("compact_support") == COMPACT_SUPPORT:
            schema_version = 4
        elif configuration.get("compact_support") == LEGACY_COMPACT_SUPPORT:
            schema_version = 3
        elif configuration.get("coordinate_normalization") == UNIT_BALL_NORMALIZATION:
            schema_version = 2
        else:
            schema_version = 1
        checkpoint = {
            "schema_version": schema_version,
            "package_version": "0.1.0",
            "field_configuration": configuration,
            "field_state": self.field.state_dict(),
            "radius": self.radius,
            "history": {name: list(values) for name, values in self.history.items()},
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }
        if schema_version < 5:
            checkpoint["time_scale"] = self._legacy_time_scale
        torch.save(checkpoint, target)

    def __repr__(self):
        return (
            "NeuralSemigroup("
            f"dimension={self.dimension}, radius={self.radius:g}, "
            f"device='{self.device}', dtype={self.dtype})"
        )
