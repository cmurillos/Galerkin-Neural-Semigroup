# Changelog

## 0.1.0 - Unreleased

- Add the compact `NeuralSemigroupProblem(basis, weak, radius)` contract.
- Keep Galerkin supervision private during target preparation and validation.
- Add a fixed autonomous `tanh` MLP with exact spectral projection.
- Expose a global Lipschitz budget independent of network depth.
- Train without trajectories on concentric radial layers built from normalized Gaussian
  directions, retaining the origin separately.
- Replace relative-angular supervision with fixed component balancing and a
  configurable complete-Jacobian term using all canonical directions at every state.
- Select checkpoints and stop from independent value validation while reporting periodic
  full-Jacobian validation audits separately.
- Add plateau-aware radial refinement that probes adjacent-interval midpoints, inserts
  the spherical layer with the largest mean error and reports radial profiles.
- Add minimum/maximum epoch budgets, tolerance, wall-clock and stalled-fit stopping
  criteria with explicit stop metadata.
- Add adaptive Dormand--Prince 5(4), fixed-step RK4 and composition diagnostics.
- Add physical projection/reconstruction and checkpoint round trips.
