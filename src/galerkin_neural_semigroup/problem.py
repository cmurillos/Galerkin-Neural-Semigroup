"""Compact public problem definition and private Galerkin supervision."""

from math import isclose
from pathlib import Path

import torch

from ._network import _SpectralMLP
from ._sampling import reference_targets, uniform_ball
from ._validation import (
    compute_device,
    floating_dtype,
    hidden_widths,
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


def _mean_loss(field, states, targets, batch_size):
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            residual = field(states[start : start + batch_size]) - targets[
                start : start + batch_size
            ]
            total += float(residual.square().sum(dim=-1).sum().item())
    return total / len(states)


class NeuralSemigroupProblem:
    """Define and train the method without exposing the Galerkin target field.

    The normal route receives an operational ``ngfield`` basis, a complete weak
    form, and the radius of the reduced training ball. Geometry, components and
    restrictions are already carried by the basis. The normalized volume measure,
    tanh MLP, field loss and exact spectral projection are fixed by the method.
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
        lr=1e-3,
        seed=0,
        device="auto",
        dtype=None,
        verbose=False,
    ):
        """Train the fixed neural field and return its induced semigroup."""
        hidden = hidden_widths(hidden)
        lipschitz = positive_real(lipschitz, "lipschitz")
        samples = positive_integer(samples, "samples", minimum=2)
        batch_size = min(positive_integer(batch_size, "batch_size"), samples)
        epochs = positive_integer(epochs, "epochs")
        lr = positive_real(lr, "lr")
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
        train_targets = reference_targets(
            coordinate_system,
            train_states,
            time_scale=self.time_scale,
            batch_size=batch_size,
        )
        validation_targets = reference_targets(
            coordinate_system,
            validation_states,
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
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        report_every = max(1, epochs // 10)

        for epoch in range(epochs):
            field.train()
            order = torch.randperm(samples, generator=generator, device=device)
            epoch_loss = 0.0
            for start in range(0, samples, batch_size):
                indices = order[start : start + batch_size]
                residual = field(train_states[indices]) - train_targets[indices]
                loss = residual.square().sum(dim=-1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training produced a nonfinite loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().item()) * len(indices)
            epoch_loss /= samples
            field.eval()
            validation_loss = _mean_loss(
                field,
                validation_states,
                validation_targets,
                batch_size,
            )
            training_losses.append(epoch_loss)
            validation_losses.append(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone() for name, value in field.state_dict().items()
                }
            if verbose and ((epoch + 1) % report_every == 0 or epoch == 0):
                print(
                    f"epoch={epoch + 1}/{epochs} "
                    f"train={epoch_loss:.6e} validation={validation_loss:.6e}"
                )

        field.load_state_dict(best_state)
        field.eval()
        final_training_loss = _mean_loss(field, train_states, train_targets, batch_size)
        norms = field.spectral_norms()
        history = {
            "training_loss": training_losses,
            "validation_loss": validation_losses,
        }
        metrics = {
            "best_epoch": best_epoch + 1,
            "training_loss": final_training_loss,
            "validation_loss": best_loss,
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
                "measure": "normalized-volume-ball",
                "activation": "tanh",
                "spectral_projection": "exact",
                "loss": "mean-squared-euclidean-field-error",
                "autonomous": True,
            },
            "training": {
                "samples": samples,
                "validation_samples": validation_samples,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "seed": seed,
                "optimizer": "Adam",
                "target_mode": "cached",
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
        if not isclose(
            float(checkpoint.get("time_scale")), self.time_scale, rel_tol=0, abs_tol=0
        ):
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
