# AGENTS.md — Galerkin Neural Semigroup

These instructions apply throughout this repository. Follow the user's explicit
request, then `docs/design-contract.md`, executable tests, README and existing
implementation. If they differ, align all affected files in the same change.

## Ownership and first checks

- NGF owns geometry, admissible physical spaces, L²-orthonormal bases, weak-form
  assembly, private reference `G`, **indexed derivatives of `G`**, and the
  **shared Taylor integrator**. Do not implement a second integrator or a
  Jacobian-supervision path in GNS.
- GNS owns fixed uniform-volume sampling, normalized cached targets, indexed
  H^k loss, the neural network, training, local learned field, persistence and
  user-facing learned flow.
- Read `docs/design-contract.md`, README and `git status --short` before edits.
  Identify `N`, `R`, `k`, Taylor `p`, dtype/device and whether the issue belongs
  in NGF or GNS. Preserve unrelated changes.
- The exact NGF commit in `pyproject.toml` is needed because current PyPI
  versions lack these derivative/integration APIs. Advance it only with tests
  in both repos and update the short revision in training metadata.

## Mathematical invariants

For the basis `(φ_i)_{i=1}^N` and weak form, NGF supplies
`G:ℝ^N→ℝ^N`. The PDE physical setting remains L²-based and orthonormality
gives `||Σz_iφ_i||_{L²}=||z||₂` on the chosen span. The ball radius is in
reduced state coordinates, not physical geometry.

Write `x=z/R∈B_N(1)`, `g_R(x)=G(Rx)/R`, `Fθ(z)=R fθ(z/R)`. These
coordinate and physical-time relations are exact. For every multi-index,
`∂_x^αg_R(x)=R^(|α|-1)∂_z^αG(Rx)`; in particular values carry `1/R`,
first derivatives carry `1`, second derivatives carry `R`. Never add a time
scale or a normalization factor to the mathematical loss.

For integer `k≥0`, the sole objective is the unit-ball H^k field loss:

```text
∫_{B_N(1)} Σ_{|α|≤k} ||∂^α fθ(x)-∂^α g_R(x)||₂² dx.
```

Implementation samples normalized volume and computes the **average over
states**, **sum over indexed derivatives** and **sum over output components**.
Each mixed multi-index appears once. This sample average is the continuous
integral divided by the ball volume. Do not insert derivative weights,
Jacobian terms, trajectories, PDE energies, boundary penalties, radial
sampling or adaptive training without an explicit method change.

`H^k` refers to regularity with respect to state coordinates. Do not claim
spatial H^k regularity of a PDE solution. Assume or verify suitable derivatives
of `G` for each problem; finite AD at sample points is not an analytic proof.
The Taylor integration order `p` is independent of `k`.

## Training and field contract

`NeuralSemigroupProblem(basis=..., weak=..., radius=R, sobolev_order=k,
quadrature=...)` is the public problem definition. `from_galerkin` supports
explicit historical/general NGF problems while keeping the reference private.
The original coordinate API exports `NeuralSemigroupProblem` and
`NeuralSemigroup`. The D-009 function-valued API also exports `System`,
`Model`, NGF `State`/`Function`/`Solution` and NGF geometry, space, restrictions
and weak-form vocabulary. Examples must not construct or expose the reference
evaluator.

The normalized network is an autonomous `tanh` MLP with free affine weights.
Do not add spectral normalization, manual/automatic budgets or Lipschitz
penalties to new training. Adam uses one fixed volume sample and distinct fixed
validation; restore the best epoch. Regenerate samples that round to the
boundary in either unit or physical coordinates. Targets must be differentiated
at `Rx`, scaled by `R^(|α|-1)`, checked, then detached and cached. Neural state
derivatives retain parameter gradients. Eval weights are detached while input
state derivatives remain available. The product of layer spectral norms is an
a posteriori bound for each fixed network inside the ball, not a uniform bound
across models or on the discontinuous exterior convention.

`semigroup.field` and `velocity` remain defined only on the open `B_N(R)`.
`velocity_zero_extended` is separate: exactly zero for `||z||>=R`, with no
smoothing, projection, continuation or freezing. The shared solver records the
first numerical contact of the accepted Taylor polynomial in `DomainExitError`;
exit event/time/mask are separate from the last interior point and completed
outputs. Never turn an event into an admissible `State`. Identity, composition
and inverse require all local domains to be admissible. Comparisons end at
first exit of either field, in the requested time direction.

Both direct `G.solve` and neural `semigroup.solve` call NGF
`integrate_field`, using the same Taylor `order=p` and matched `step` or
`tolerance` settings for trajectory comparisons. `G.solve(..., radius=R)`
can impose the same domain. `defect` is a numerical composition diagnostic,
not a loss or a PDE error. Explicit high-order AD can be expensive and stiff
fields may require a different problem-specific solver study.

All state tensors have shape `[...,N]`, with arbitrary batch axes. Preserve
strict device/dtype, finite-value and shape checks. Inputs may require state
gradients; do not detach them for convenience. Avoid per-state Python loops.
Independent trajectory evaluation accepts strictly monotone time grids in
either direction, including negative times. Fixed reference targets may be
cached on CPU for large GPU studies; batches must return to the model device
without changing samples, indexed weights or parameter gradients. A training
batch may reuse the affine weights without projection, but the cached graph must not
survive an optimizer step.

## Compatibility and reproducibility

Schema 6 saves free weights, exterior-zero convention, the local-field marker, `k`, radius, basis signature, model
state, history and metadata. It contains neither the weak-form callable nor
numerical reference. Loading needs a compatible problem. Historical schemas 1–5 retain spectral projection; 1–4 also retain their past
normalization, taper and time factors. Never silently reinterpret them as schema 6. Use `torch.load(..., weights_only=True)`.

Public docs and code are English. Update README, design contract, example,
changelog and tests for method changes. Keep the package-root API compact;
private modules do not belong in end-user examples. For a scientific
experiment, separate projection, direct Galerkin, learned-field and integration
errors. Negative results are valid. Never adjust sampling or hide warnings to
make an experiment look favorable.

## Files and verification

- `problem.py`: private NGF reference, targets, H^k training and loading.
- `_sampling.py`: normalized volume and scaled indexed target preparation.
- `_network.py`: free-weight MLP, local wrapper, zero convention and legacy wrappers.
- `_lipschitz.py`: retained sampled derivative diagnostic; never a training budget.
- `semigroup.py`: public learned flow and versioned checkpoints.
- `_evaluation.py`: independent normalized field and trajectory diagnostics;
  obtain reference derivatives, integral weights and integration from NGF.
- `workflow.py`: function-valued study and learned flow, NGF functional types,
  indexed derivative option, grouped independent evaluation reports.
- `docs/design-contract.md`: mathematical contract and limitations.

For simultaneous sibling checkouts, install NGF from its local source and GNS
without resolving a second dependency copy:

```bash
python -m pip install -e "../Numerical-Galerkin-Field[dev]"
python -m pip install -e . --no-deps
python -m pytest
ruff check .
ruff format --check .
python -m build
```

Test exact radius derivative scaling (including a mixed derivative), gradients
through H^k loss, open-ball failure, matched Galerkin/neural integrator calls,
checkpoint schema 6 and legacy schemas 1–5. Include an end-to-end reduced
training smoke test when changing user workflow. Report any untested device,
stiffness or PDE examples accurately.
