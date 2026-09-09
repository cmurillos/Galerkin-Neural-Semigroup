# Changelog

## 0.1.0 - Unreleased

- Add the compact `NeuralSemigroupProblem(basis, weak, radius)` contract.
- Keep Galerkin supervision private during target preparation and validation.
- Add a fixed autonomous `tanh` MLP with exact spectral projection.
- Expose a global Lipschitz budget independent of network depth.
- Train on normalized-volume samples of the reduced ball without trajectories.
- Replace the unnormalized field MSE with a regularized relative error plus a
  configurable angular-alignment term.
- Report total, relative and angular training and validation losses separately.
- Add adaptive Dormand--Prince 5(4), fixed-step RK4 and composition diagnostics.
- Add physical projection/reconstruction and checkpoint round trips.
