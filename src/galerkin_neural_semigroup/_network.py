"""The fixed autonomous MLP used by the method."""

from math import prod, sqrt

import torch
from torch import nn
from torch.nn import functional as functional

from ._validation import hidden_widths, positive_integer, positive_real

UNIT_BALL_NORMALIZATION = "unit-ball"
COMPACT_SUPPORT = "smooth-annulus"
LEGACY_COMPACT_SUPPORT = "smooth-unit-ball"
SUPPORT_START_RADIUS = 1.0
SUPPORT_END_RADIUS = 2.0
LEGACY_SUPPORT_START_RADIUS = 0.9
LEGACY_SUPPORT_END_RADIUS = 1.0


class _SpectralMLP(nn.Module):
    """A tanh MLP whose affine weights are projected onto spectral balls."""

    def __init__(self, dimension, hidden, lipschitz, *, device, dtype):
        super().__init__()
        self.dimension = positive_integer(dimension, "dimension")
        self.hidden = hidden_widths(hidden)
        self.lipschitz_bound = positive_real(lipschitz, "lipschitz")
        widths = (self.dimension, *self.hidden, self.dimension)
        self.depth = len(widths) - 1
        self.layer_bound = self.lipschitz_bound ** (1.0 / self.depth)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self._evaluation_weights = None
        self._evaluation_biases = None
        for input_width, output_width in zip(widths[:-1], widths[1:]):
            weight = torch.empty(output_width, input_width, device=device, dtype=dtype)
            bias = torch.empty(output_width, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(weight, a=sqrt(5))
            bound = 1 / sqrt(input_width)
            nn.init.uniform_(bias, -bound, bound)
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(bias))

    @property
    def device(self):
        return self.weights[0].device

    @property
    def dtype(self):
        return self.weights[0].dtype

    def _states(self, states):
        if not isinstance(states, torch.Tensor):
            raise TypeError("states must be a torch tensor.")
        if states.ndim < 1 or states.shape[-1] != self.dimension:
            raise ValueError(f"states must have shape [...,{self.dimension}].")
        if states.device != self.device or states.dtype != self.dtype:
            raise ValueError("states and the neural field must share device and dtype.")

    def _project(self, weight):
        norm = torch.linalg.matrix_norm(weight, ord=2)
        scale = torch.clamp(norm / self.layer_bound, min=1.0)
        return weight / scale

    def effective_weights(self):
        if not self.training:
            if self._evaluation_weights is None:
                with torch.no_grad():
                    self._evaluation_weights = tuple(
                        self._project(weight).detach() for weight in self.weights
                    )
            return self._evaluation_weights
        return tuple(self._project(weight) for weight in self.weights)

    def effective_biases(self):
        if not self.training:
            if self._evaluation_biases is None:
                self._evaluation_biases = tuple(bias.detach() for bias in self.biases)
            return self._evaluation_biases
        return tuple(self.biases)

    def _clear_evaluation_cache(self):
        self._evaluation_weights = None
        self._evaluation_biases = None

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise TypeError("mode must be a boolean.")
        self._clear_evaluation_cache()
        return super().train(mode)

    def _apply(self, function):
        self._clear_evaluation_cache()
        return super()._apply(function)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._clear_evaluation_cache()
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        return result

    @torch.no_grad()
    def spectral_norms(self):
        return tuple(
            float(torch.linalg.matrix_norm(weight, ord=2).item())
            for weight in self.effective_weights()
        )

    @torch.no_grad()
    def effective_lipschitz_bound(self):
        return prod(self.spectral_norms())

    def _forward_validated(self, states):
        value = states
        weights = self.effective_weights()
        biases = self.effective_biases()
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            value = functional.linear(value, weight, bias)
            if index + 1 < self.depth:
                value = torch.tanh(value)
        return value

    def forward(self, states):
        self._states(states)
        return self._forward_validated(states)

    def configuration(self):
        return {
            "dimension": self.dimension,
            "hidden": list(self.hidden),
            "lipschitz": self.lipschitz_bound,
        }


class _UnitBallField(nn.Module):
    """Expose a unit-ball network in physical coordinates with optional support."""

    def __init__(
        self,
        core,
        radius,
        *,
        compact_support=False,
        support_start_radius=SUPPORT_START_RADIUS,
        support_end_radius=SUPPORT_END_RADIUS,
    ):
        super().__init__()
        if not isinstance(core, _SpectralMLP):
            raise TypeError("core must be a spectral MLP.")
        self.core = core
        self.radius = positive_real(radius, "radius")
        if not isinstance(compact_support, bool):
            raise TypeError("compact_support must be a boolean.")
        self.compact_support = compact_support
        self.support_start_radius = positive_real(support_start_radius, "support_start_radius")
        self.support_end_radius = positive_real(support_end_radius, "support_end_radius")
        if self.support_start_radius >= self.support_end_radius:
            raise ValueError("support_start_radius must be less than support_end_radius.")

    @property
    def dimension(self):
        return self.core.dimension

    @property
    def device(self):
        return self.core.device

    @property
    def dtype(self):
        return self.core.dtype

    def _states(self, states):
        self.core._states(states)

    def normalized(self, states):
        """Evaluate the deployed field in normalized coordinates."""
        self.core._states(states)
        return self._normalized_validated(states)

    def _normalized_validated(self, states):
        if not self.compact_support:
            return self.core._forward_validated(states)
        flat = states.reshape(-1, self.dimension)
        radii = torch.linalg.vector_norm(flat, dim=-1)
        # Preserve nonfinite values for the integrator's existing error checks.
        active = (radii < self.support_end_radius) | ~torch.isfinite(radii)
        inside = flat[active]
        r = radii[active]
        transition = (
            (r - self.support_start_radius) / (self.support_end_radius - self.support_start_radius)
        ).clamp(0, 1)
        taper = 1.0 - 3.0 * transition.square() + 2.0 * transition.pow(3)
        # Outside the training ball, only evaluate the network on its boundary.
        projected = inside / r.clamp_min(1.0).unsqueeze(-1)
        values = self.core._forward_validated(projected) * taper.unsqueeze(-1)
        result = torch.zeros_like(flat)
        result[active] = values
        return result.reshape(states.shape)

    def _forward_validated(self, states):
        return self.radius * self._normalized_validated(states / self.radius)

    def forward(self, states):
        """Evaluate the scaled, optionally supported field in physical coordinates."""
        self._states(states)
        return self._forward_validated(states)

    def spectral_norms(self):
        return self.core.spectral_norms()

    def effective_lipschitz_bound(self):
        base_bound = self.core.effective_lipschitz_bound()
        if not self.compact_support:
            return base_bound
        with torch.no_grad():
            zero = torch.zeros(self.dimension, device=self.device, dtype=self.dtype)
            offset = float(torch.linalg.vector_norm(self.core._forward_validated(zero)).item())
        # Radial projection onto the unit ball is 1-Lipschitz. On the unit
        # ball |core(x)| <= |core(0)| + Lip(core); scaling by R cancels.
        return base_bound + 1.5 * (offset + base_bound) / (
            self.support_end_radius - self.support_start_radius
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.core._clear_evaluation_cache()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def configuration(self):
        configuration = self.core.configuration()
        configuration["coordinate_normalization"] = UNIT_BALL_NORMALIZATION
        if self.compact_support:
            if (
                self.support_start_radius == LEGACY_SUPPORT_START_RADIUS
                and self.support_end_radius == LEGACY_SUPPORT_END_RADIUS
            ):
                configuration["compact_support"] = LEGACY_COMPACT_SUPPORT
                configuration["support_inner_radius_fraction"] = LEGACY_SUPPORT_START_RADIUS
            else:
                configuration["compact_support"] = COMPACT_SUPPORT
                configuration["support_start_radius_fraction"] = self.support_start_radius
                configuration["support_end_radius_fraction"] = self.support_end_radius
        return configuration
