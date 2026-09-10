# Changelog

## 0.1.0 - Unreleased

- Add the compact `NeuralSemigroupProblem(basis, weak, radius)` contract.
- Keep Galerkin supervision private during target preparation and validation.
- Add a fixed autonomous `tanh` MLP with exact spectral projection.
- Expose a global Lipschitz budget independent of network depth.
- Train on normalized-volume samples of the reduced ball without trajectories.
- Replace relative-angular supervision with fixed component balancing and a
  configurable Jacobian-vector-product term.
- Report total, value and Jacobian training and validation losses separately.
- Add plateau-aware error refinement with radially stratified probes and local
  error-weighted kernel sampling.
- Add minimum/maximum epoch budgets, tolerance, wall-clock and stalled-fit stopping
  criteria with explicit stop metadata.
- Add adaptive Dormand--Prince 5(4), fixed-step RK4 and composition diagnostics.
- Add physical projection/reconstruction and checkpoint round trips.
