"""A trained autonomous neural field and its induced flow."""

from math import isfinite
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import torch

from ._integration import solve as integrate
from ._network import UNIT_BALL_NORMALIZATION
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
        time_scale,
        history=None,
        metrics=None,
        metadata=None,
    ):
        self.field = field
        self._coordinate_system = coordinate_system
        self.radius = float(radius)
        self.time_scale = float(time_scale)
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
        return self.field(states) / self.time_scale

    def solve(self, z0, times, *, step=None, tolerance=None):
        """Evolve reduced coordinates from the first requested physical time.

        Omitting ``step`` uses adaptive Dormand--Prince 5(4). Providing ``step``
        selects fixed-step RK4. ``step`` is expressed in physical time.
        """
        if not isinstance(z0, torch.Tensor):
            z0 = torch.as_tensor(z0, device=self.device, dtype=self.dtype)
        if not isinstance(times, torch.Tensor):
            times = torch.as_tensor(times, device=self.device, dtype=self.dtype)
        scaled_times = times / self.time_scale
        scaled_step = None if step is None else positive_real(step, "step") / self.time_scale
        return integrate(
            self.field,
            z0,
            scaled_times,
            step=scaled_step,
            tolerance=tolerance,
        )

    def __call__(self, state, time, *, step=None, tolerance=None):
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
        return self.solve(state, times, step=step, tolerance=tolerance)[-1]

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
    ):
        """Project, evolve and reconstruct a physical initial state."""
        z0 = self.project(initial_state, quadrature=projection_quadrature)
        states = self.solve(z0, times, step=step, tolerance=tolerance)
        return self.reconstruct(states, points, cells=cells, boundary=boundary)

    def defect(self, state, first_time, second_time, *, step=None, tolerance=None):
        """Return Psi_t(Psi_s(z)) - Psi_{t+s}(z) for integrator diagnostics."""
        after_first = self(state, first_time, step=step, tolerance=tolerance)
        composed = self(after_first, second_time, step=step, tolerance=tolerance)
        direct = self(
            state,
            float(first_time) + float(second_time),
            step=step,
            tolerance=tolerance,
        )
        return composed - direct

    def save(self, path):
        """Save network parameters and reproducibility metadata, but not the oracle."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        configuration = self.field.configuration()
        schema_version = (
            2 if configuration.get("coordinate_normalization") == UNIT_BALL_NORMALIZATION else 1
        )
        checkpoint = {
            "schema_version": schema_version,
            "package_version": "0.1.0",
            "field_configuration": configuration,
            "field_state": self.field.state_dict(),
            "radius": self.radius,
            "time_scale": self.time_scale,
            "history": {name: list(values) for name, values in self.history.items()},
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }
        torch.save(checkpoint, target)

    def __repr__(self):
        return (
            "NeuralSemigroup("
            f"dimension={self.dimension}, radius={self.radius:g}, "
            f"time_scale={self.time_scale:g}, device='{self.device}', dtype={self.dtype})"
        )
