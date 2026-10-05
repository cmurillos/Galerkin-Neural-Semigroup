"""Function-valued neural flow with the same physical-state API as NGF."""

from types import MappingProxyType

import torch
from ngfield import FunctionalFlow
from ngfield import System as NumericalSystem

from ._network import OPEN_BALL_DOMAIN
from .problem import NeuralSemigroupProblem


class System:
    """Specify one neural study on a fixed NGF basis and an open coordinate ball.

    The training radius and Sobolev order are properties of this study. The
    trained model evolves physical functions; its reduced coordinates remain
    an explicit interoperability option.
    """

    def __init__(self, *, basis, weak, radius, quadrature=None, sobolev_order=1):
        self._problem = NeuralSemigroupProblem(
            basis=basis,
            weak=weak,
            radius=radius,
            quadrature=quadrature,
            sobolev_order=sobolev_order,
        )

    @property
    def basis(self):
        return self._problem.basis

    @property
    def space(self):
        return getattr(self.basis, "space", None)

    @property
    def geometry(self):
        return self.basis.geometry

    @property
    def dimension(self):
        return self._problem.dimension

    @property
    def radius(self):
        return self._problem.radius

    @property
    def sobolev_order(self):
        return self._problem.sobolev_order

    def train(
        self,
        *,
        hidden,
        samples=10_000,
        batch_size=256,
        epochs=1_000,
        lr=1e-3,
        seed=0,
        device="auto",
        dtype=None,
        verbose=False,
    ):
        """Train the H^k field loss and return a function-valued learned model."""
        return Model(
            self._problem.train(
                hidden=hidden,
                samples=samples,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                seed=seed,
                device=device,
                dtype=dtype,
                verbose=verbose,
            )
        )

    def load(self, path, *, device="auto", dtype=None):
        """Load a matching local model (schemas 5 and 6) into the function-valued workflow."""
        return Model(self._problem.load(path, device=device, dtype=dtype))


class Model(NumericalSystem):
    """Learned local flow in V_N; training and the Galerkin reference stay private."""

    def __init__(self, neural):
        if neural.field.configuration().get("domain") != OPEN_BALL_DOMAIN:
            raise ValueError(
                "The function-valued workflow requires an open-ball schema-5 or schema-6 model."
            )
        self._neural = neural
        # Keep the public NumericalSystem relationship while initializing the
        # compatible-flow contract without rebuilding the Galerkin field.
        FunctionalFlow.__init__(self, neural._coordinate_system, neural.field, radius=neural.radius)
        self.evaluate = Evaluation(self)

    @property
    def training_history(self):
        return self._neural.history

    @property
    def training_metrics(self):
        return self._neural.metrics

    @property
    def metadata(self):
        return self._neural.metadata

    def velocity_zero_extended(self, coefficients):
        """Exterior-zero convention on coordinate tensors; not a phase velocity."""
        return self._neural.velocity_zero_extended(coefficients)

    def save(self, path):
        self._neural.save(path)


class FieldReport:
    """Independent normalized field statistics on reserved physical states."""

    def __init__(self, data):
        self._data = MappingProxyType(data)

    def raw(self):
        return dict(self._data)

    def rmse(self):
        return self._data["field_rmse"]

    def derivatives(self, *, by_order=True):
        """RMS for orders 1..k, or their aggregate; the field value is order 0."""
        values = self._data["derivative_rmse_by_order"][1:]
        return values if by_order else torch.linalg.vector_norm(values)

    def sobolev(self):
        return self._data["sobolev_rmse"]

    def radial(self):
        return {
            "edges": self._data["radial_edges"],
            "counts": self._data["radial_counts"],
            "rmse": self._data["radial_rmse"],
        }

    def lipschitz(self):
        """Two sampled maxima and the separate spectral bound on the learned field."""
        return {
            "reference_sample": self._data["reference_lipschitz_sample"],
            "learned_sample": self._data["learned_lipschitz_sample"],
            "learned_bound": self._data["learned_lipschitz_bound"],
        }


class TrajectoryReport:
    """Normalized comparison with the independent numerical Galerkin flow."""

    def __init__(self, data):
        self._data = MappingProxyType(data)

    def raw(self):
        return dict(self._data)

    def trajectories(self):
        return {key: self._data[key] for key in ("times", "reference_states", "learned_states")}

    def error(self):
        return {
            "per_path": self._data["trajectory_error"],
            "mean": self._data["trajectory_error_mean"],
            "std": self._data["trajectory_error_std"],
            "common_count": self._data["common_count"],
        }

    def field_on_paths(self):
        return self._data["visited_field_rmse"]

    def integrals(self):
        return {
            "gap_mean": self._data["mass_gap_mean"],
            "change_reference": self._data["mass_change_reference"],
            "change_learned": self._data["mass_change_learned"],
            "rate_gap_mean": self._data["mass_rate_gap_mean"],
        }

    def norms(self):
        return {
            "gap_mean": self._data["norm_gap_mean"],
            "change_reference": self._data["norm_change_reference"],
            "change_learned": self._data["norm_change_learned"],
            "radial_rate_reference": self._data["radial_rate_reference"],
            "radial_rate_learned": self._data["radial_rate_learned"],
            "radial_rate_gap_mean": self._data["radial_rate_gap_mean"],
        }

    def exits(self):
        """Numerical exits, survival fractions and count of common survivors."""
        return {
            key: self._data[key]
            for key in (
                "reference_exit_time",
                "learned_exit_time",
                "comparison_exit_time",
                "reference_exit_state",
                "learned_exit_state",
                "reference_last_accepted_time",
                "learned_last_accepted_time",
                "reference_survival",
                "learned_survival",
                "common_count",
            )
        }

    def time_refinement(self):
        if "reference_refinement" not in self._data:
            raise ValueError("Request refine=True when evaluating trajectories.")
        return {
            key: self._data[key]
            for key in (
                "reference_refinement",
                "learned_refinement",
                "reference_refinement_mean",
                "learned_refinement_mean",
                "reference_refinement_count",
                "learned_refinement_count",
            )
        }


class Evaluation:
    """Diagnostics on reserved data; never part of the training loss."""

    def __init__(self, model):
        self._model = model

    def field(self, states_test, *, radial_edges=None, batch_size=128):
        states = self._model._sample_coordinates(states_test)
        return FieldReport(
            self._model._neural.evaluate_field(
                states, radial_edges=radial_edges, batch_size=batch_size
            )
        )

    def trajectories(
        self, initial_states_test, times, *, step=None, tolerance=None, order=4, refine=False
    ):
        states = self._model._sample_coordinates(initial_states_test)
        return TrajectoryReport(
            self._model._neural.evaluate_trajectories(
                states, times, step=step, tolerance=tolerance, order=order, refine=refine
            )
        )


__all__ = ["Evaluation", "FieldReport", "Model", "System", "TrajectoryReport"]
