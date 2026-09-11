# Changelog

## 0.1.0 - Unreleased

- Add the compact `NeuralSemigroupProblem(basis, weak, radius)` contract.
- Keep Galerkin supervision private during target preparation and validation.
- Add a fixed autonomous `tanh` MLP with exact spectral projection.
- Expose a global Lipschitz budget independent of network depth.
- Restore the direct squared field-matching objective `F_theta - G`.
- Add two fixed vectorized sampling modes: normalized volume and uniform radius.
- Normalize network coordinates to the unit ball while preserving the physical reduced
  API through `F(z) = R * F_hat(z/R)`.
- Keep training non-adaptive and cache only Galerkin field values.
- Add adaptive Dormand--Prince 5(4), fixed-step RK4 and composition diagnostics.
- Add physical projection/reconstruction and checkpoint round trips.
