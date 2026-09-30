# Changelog

## Unreleased

- Add a function-valued `System`/`Model` workflow with the same NGF basis,
  state, spatial evaluation and indexed coordinate derivative vocabulary.
  Group independent field and trajectory reports behind explicit methods,
  while preserving the coordinate and checkpoint APIs.
- Add independent, normalized field and trajectory evaluation against the
  private NGF reference, including derivative/radial profiles, sampled
  Lipschitz diagnostics, per-component integrals, L² rates, numerical ball
  exit/survival and optional matched temporal refinement. Keep these metrics
  separate from the training objective and persisted model metrics.

- Train the normalized reduced field with the unweighted, indexed H^k loss;
  source Galerkin state derivatives from NGF and cache scaled targets.
- Use fixed normalized-volume sampling and an open-ball local field for new
  models. Remove the time-scale, exterior taper and radial training modes from
  new training; older checkpoint field semantics remain loadable.
- Call the NGF Taylor-jet integrator for both Galerkin and neural fields, with
  matching order and time-step controls. Store new models in checkpoint schema 5.
- Pin the NGF source commit containing indexed derivatives and the shared solver.

### Earlier unreleased changes retained for historical context

- Preserve the raw learned field through the full training radius R; taper
  from R to 2R, projecting network inputs onto the training ball beyond R.
  The field becomes zero from 2R onward. The deployed Lipschitz bound can
  still exceed the raw network budget.
- Estimate the default spectral budget from reproducible private Galerkin
  difference quotients, with a configurable multiplier and an explicit
  uncertified status; preserve numeric manual budgets.
- Save the new taper in schema-4 checkpoints, retaining prior checkpoint
  behavior for schemas 1, 2, and 3.
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
