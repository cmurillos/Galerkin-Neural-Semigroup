# Changelog

## Unreleased

- Add a fixed 50/50 mixture of volume and uniform-radius sampling.
- Apply a smooth compact-support taper after training to new neural fields:
  they agree with the raw network through radius 0.9R and vanish at and outside R.
  Keep the raw loss and report the deployed validation error and its larger
  global Lipschitz bound separately.
- Store the taper in schema-3 checkpoints; schema-1 and schema-2 checkpoints
  retain their original behavior.

## 0.1.0 - Unreleased

- Add a root `AGENTS.md` with method-specific API, experiment, testing and cross-package
  instructions for coding agents, and link it from the README.
- Add the compact `NeuralSemigroupProblem(basis, weak, radius)` contract.
- Keep Galerkin supervision private during target preparation and validation.
- Add a fixed autonomous `tanh` MLP with exact spectral projection.
- Expose a global Lipschitz budget independent of network depth.
- Restore the direct squared field-matching objective `F_theta - G`.
- Add two fixed vectorized sampling modes: normalized volume and uniform radius.
- Normalize network coordinates to the unit ball while preserving the physical reduced
  API through `F(z) = R * F_hat(z/R)`.
- Keep training non-adaptive and cache only Galerkin field values.
- Remove repeated device synchronizations during training and Runge--Kutta stages,
  preallocate cached targets and avoid parameter graphs during ordinary evaluation.
- Add adaptive Dormand--Prince 5(4), fixed-step RK4 and composition diagnostics.
- Add physical projection/reconstruction and checkpoint round trips.
