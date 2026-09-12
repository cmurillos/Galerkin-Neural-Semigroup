# AGENTS.md

## Purpose and scope

This file gives coding agents the repository-specific instructions needed to use,
explain, test, and modify Galerkin Neural Semigroup (GNS) without changing the method by
accident. It applies to the entire repository.

GNS is the neural layer of a two-package project:

- **Numerical Galerkin Field (`ngfield`)** owns geometry, admissible spaces, fixed
  L2-orthonormal bases, weak-form assembly, the numerical Galerkin field, physical
  projection/reconstruction, and direct Galerkin time evolution.
- **Galerkin Neural Semigroup (`galerkin_neural_semigroup`)** owns fixed reduced-ball
  sampling, cached supervision, the spectrally projected neural field, training,
  persistence, and the learned continuous-flow interface.

Keep that separation strict. Users describe a weak problem and basis; GNS constructs the
numerical Galerkin reference privately. Public examples and notebooks must not ask users to
construct `G`, import GNS internals, or reimplement the network, loss, sampler, optimizer, or
integrator.

## Instruction and source-of-truth order

Follow, in order:

1. The user's explicit request.
2. Accepted decisions D-001 through D-008 in `docs/design-contract.md`.
3. Executable behavior in `tests/`.
4. The public contract and examples in `README.md` and `examples/heat.py`.
5. Existing implementation details.

If sources disagree, do not silently reinterpret the method. Identify the mismatch and keep
design contract, implementation, tests, README, example, and changelog synchronized. A
change to an accepted fixed choice—architecture, loss, sampling law, normalization,
optimizer, or reference visibility—is a method/design change, not a routine refactor.

The package is alpha research software at version `0.1.0`. Its `ngfield` dependency is pinned
in `pyproject.toml` to an exact source commit because required post-0.9.0 APIs are not yet in
the published Numerical Galerkin Field 0.9.0 release. Do not replace the pin with an
incompatible PyPI constraint or advance it without cross-repository verification.

## First actions for every task

Before editing:

1. Read `README.md` and all of `docs/design-contract.md`.
2. Inspect `git status --short`; preserve unrelated user changes.
3. Locate current signatures and tests with `rg`; do not infer an API from an old notebook.
4. Write down the coordinate convention (`z`, `x=z/R`), radius `R`, time scale `tau`, and
   tensor shapes involved in the task.
5. Decide whether the issue belongs to GNS or its `ngfield` dependency. Fix it in the owning
   package rather than duplicating behavior.
6. Reproduce defects with the smallest deterministic test before changing code.

Ask a question only when the missing information changes the scientific problem or a public
method choice. Do not make users select private helper classes or expose the reference field
to work around missing context.

## Method in one page

Let `basis = (phi_1, ..., phi_N)` be a fixed real L2-orthonormal operational basis prepared
by `ngfield`, and let

```text
G: R^N -> R^N,
G_i(z) = a(Phi(z); phi_i)
```

be the numerical Galerkin field generated from the complete autonomous weak form. GNS learns
this **field**, not isolated solution trajectories and not a map with time as an input.

The physical training domain is the reduced coordinate ball

```text
B_N(R) = {z in R^N : ||z||_2 < R}.
```

Because the basis is L2-orthonormal, `R` is also the L2 radius of reconstructed reduced
states. It is not a spatial/geometric radius and not a time horizon.

The network always uses normalized coordinates

```text
x = z / R in B_N(1).
```

For `time_scale = tau`, training targets and public fields obey

```text
G_hat(x)       = (tau / R) * G(R*x),
F(z)           = R * F_hat(z/R)       ~= tau * G(z),
velocity(z)    = F(z) / tau            ~= G(z).
```

`semigroup.field(z)` returns `F(z)`, the scaled learned field in original reduced
coordinates. `semigroup.velocity(z)` returns the physical-time velocity. `solve` receives
physical times and divides them by `tau` internally. Never drop or duplicate an `R` or `tau`
factor; test all three equations together whenever scaling code changes.

The exact continuous flow induced by an autonomous globally Lipschitz neural field satisfies
identity and composition by construction. The package numerically approximates that flow.
It does not train a semigroup-composition loss. `semigroup.defect(...)` measures numerical
integration/roundoff defect, not a learned structural penalty.

## Canonical public workflow

Use this route in new examples and user-facing answers:

```python
import torch

from galerkin_neural_semigroup import NeuralSemigroupProblem
from ngfield import SimplicialDomain, Space, ZeroTrace, grad, inner

geometry = SimplicialDomain(
    vertices=vertices,
    simplices=simplices,
    regions=regions,
    boundaries=boundaries,
)
V = Space(
    geometry=geometry,
    components=1,
    regularity=1,
    restrictions=[ZeroTrace(component=0, boundary="all")],
)
basis = V.basis("laplacian", size=N, degree=1)


def weak(u, v, dx, ds):
    return -kappa * inner(grad(u[0]), grad(v[0])) * dx


problem = NeuralSemigroupProblem(
    basis=basis,
    weak=weak,
    radius=R,
    quadrature=None,
    time_scale=1.0,
)

semigroup = problem.train(
    hidden=(64, 64),
    lipschitz=L,
    samples=10_000,
    sampling="volume",
    batch_size=256,
    epochs=1_000,
    lr=1e-3,
    seed=0,
)

z0 = semigroup.project(initial_state)
times = torch.linspace(t0, t1, steps, dtype=semigroup.dtype, device=semigroup.device)
Z = semigroup.solve(z0, times)
U = semigroup.reconstruct(Z, points)
```

A normal GNS user-facing program needs `NeuralSemigroupProblem` plus public `ngfield`
geometry/space/form symbols. It does not import or construct `GalerkinField`; the numerical
field remains an internal coordinate system built by the problem.

For a general basis outside the modern `Space` route, use the explicit compatibility adapter:

```python
problem = NeuralSemigroupProblem.from_galerkin(
    galerkin_problem,
    basis=basis,
    radius=R,
    quadrature=None,
    time_scale=1.0,
)
```

This adapter also keeps the reference field private.

## Public API contract

Only two names are exported from the package root:

- `NeuralSemigroupProblem`: untrained problem definition and training/loading entry point.
- `NeuralSemigroup`: trained result type; users receive it from `train` or `load` rather than
  constructing it directly.

### `NeuralSemigroupProblem`

The normal constructor receives keyword-only `basis`, `weak`, `radius`, optional
`quadrature`, and optional positive `time_scale`. Geometry, physical value shape,
components, and homogeneous restrictions are already attached to the operational basis and
must not be repeated.

`problem.train(...)` exposes choices that the present method leaves to the user:

- `hidden`: tuple/list of positive hidden widths; `()` is the exact affine test architecture;
- `lipschitz`: positive global spectral budget;
- `samples`, `batch_size`, `epochs`, `lr`, and `seed`;
- `sampling`: exactly `"volume"` or `"radius"`;
- advanced `device` and `dtype` overrides;
- `verbose` progress output.

Adam, the direct field loss, tanh activations, exact spectral projection, fixed cached
targets, independent fixed validation data, and unit-ball normalization are method choices,
not exposed switches.

`problem.load(path, ...)` reconstructs a checkpoint against the same problem definition and
ordered basis. It validates recorded compatibility data but cannot prove that two arbitrary
bases with the same coarse signature contain identical functions.

### `NeuralSemigroup`

The trained object provides:

| Member | Meaning |
| --- | --- |
| `field(z)` | Scaled learned field `F(z) ~= tau G(z)` in physical reduced coordinates. |
| `velocity(z)` | Learned physical-time velocity `F(z)/tau`. |
| `solve(z0, times, ...)` | Trajectory at physical times. |
| `semigroup(z, time, ...)` | Apply the numerical learned flow from time zero to one time. |
| `project(source, ...)` | Delegate physical L2 projection to the fixed `ngfield` coordinate system. |
| `reconstruct(states, points, ...)` | Delegate synthesis to the same fixed basis. |
| `solve_physical(...)` | Project, evolve, and reconstruct in one call. |
| `defect(z,s,t,...)` | Numerical composition diagnostic. |
| `save(path)` | Store network/configuration/history/metadata, not the reference evaluator. |
| `dimension`, `device`, `dtype`, `basis`, `space`, `geometry` | Read-only coordinate context. |
| `history`, `metrics`, `metadata` | Read-only mappings for training and reproducibility. |

Do not expand the root export surface with private architecture, sampler, loss, or integrator
classes. A public addition requires an explicit design decision, documentation, validation,
and compatibility plan.

## Sampling and dataset invariants

Training supports exactly two vectorized probability measures. With independent
`s ~ Uniform(0,1)` and Gaussian direction `xi`:

```text
direction = xi / ||xi||_2
sampling="volume": x = s^(1/N) * direction
sampling="radius": x = s       * direction
physical target state: z = R*x
```

`"volume"` is normalized Lebesgue volume on the N-ball. `"radius"` is uniform radius
times uniform angular measure; it is not uniform volume. Preserve this distinction in code,
plots, docs, and scientific interpretation.

Training states and validation states are drawn once, independently, from the same selected
measure. They stay fixed for the entire run. Epochs shuffle indices only. Do not resample per
epoch, adapt points, add radial layers, mix the two measures, or change validation measure
without an approved method change.

Targets are evaluated at `R*x` in vectorized batches, scaled by `tau/R`, detached, checked
for finiteness, and cached. Target generation must not retain an autograd graph through the
Galerkin reference. The implementation must avoid Python loops over individual states.

The validation size is currently deterministic:

```text
max(1, min(4096, samples // 10)).
```

The best independent validation state is restored after training; the returned network is
not merely the last epoch.

## Loss invariant

The sole objective for a batch of normalized states is

```text
L_B(theta) = (1/B) * sum_i ||F_hat_theta(x_i) - G_hat(x_i)||_2^2.
```

It averages over states and **sums** over reduced coordinates. Preserve this exact reduction.
There is no relative, angular, radial, trajectory, semigroup, Jacobian, JVP, conservation,
boundary, or adaptive loss term. Such quantities may be independent evaluation diagnostics,
but do not insert them into training or report them as part of the implemented method.

## Network and Lipschitz invariant

The normalized field is an autonomous affine MLP with componentwise `tanh` after every
hidden affine layer and no activation after the output. Time is never an input. For affine
weights `W_k`, evaluation uses the exact projection

```text
Wbar_k = W_k / max(1, ||W_k||_2 / gamma),
gamma = L^(1/d),
```

where `d` is the number of affine layers and `L` is the requested global budget. Spectral
norms are computed with `torch.linalg.matrix_norm(..., ord=2)`, not a finite power-iteration
estimate presented as a certificate. The product of effective layer norms must not exceed
the requested budget up to numerical tolerance.

The coordinate transform `F(z)=R*F_hat(z/R)` preserves the field's Lipschitz quotient. The
physical-time velocity has the corresponding scale `1/tau`.

During optimization, spectral projection stays in the differentiable forward path. In eval
mode, projected weights and biases are cached and detached so ordinary ODE solves do not
recompute matrix norms or build a parameter graph. Inputs requiring gradients must remain
differentiable; never detach states as an optimization. Calling `.train()` invalidates the
cache, and loading/moving parameters must do the same.

## Tensor, dtype, and device contract

The final axis is always the reduced coordinate axis:

```text
field:       [...,N] -> [...,N]
velocity:    [...,N] -> [...,N]
solve:       z0:[*S,N], times:[T] -> [T,*S,N]
reconstruct: [*S,N], points:[Q,p] -> [*S,Q,*value_shape]
```

All leading state axes are arbitrary batch axes, including multiple and zero-sized axes.
Preserve exact shape instead of flattening permanently or squeezing singleton components.

`problem.train` defaults to CUDA when available and otherwise CPU; default precision is
`float64`. Advanced precision is limited to `torch.float32` and `torch.float64`. Direct
`semigroup.field` evaluation is strict about tensor dtype/device. Convenience flow methods
may convert non-tensor states/times as currently documented, but internal hot paths must not
introduce repeated host/device synchronization or silent mixed precision.

Set both PyTorch's global seed and the correct CPU/CUDA generator as the implementation does.
Record device, dtype, seed, sampling measure, training controls, normalization, and reference
quadrature metadata. Determinism metadata improves reproducibility; do not promise bitwise
identity across different PyTorch versions, hardware, or devices without evidence.

## Flow and integration contract

Omitting `step` selects adaptive Dormand--Prince 5(4). Supplying positive `step` selects
fixed-step RK4; `step` is expressed in physical time and acts as a maximum internal step.
`step` and `tolerance` cannot be supplied together. Requested times must be finite,
one-dimensional, nonempty, and strictly monotone unless only one time is requested. Forward
and backward time are supported.

Integration preserves autograd with respect to an initial state that explicitly requires a
gradient. Adaptive step accept/reject decisions are discrete and are not claimed to be
differentiable. Keep finite-state checks and the internal step budget. Do not clip exploding
states, suppress failures, or call explicit RK methods robust for stiff fields without a
problem-specific study.

`defect(z,s,t)` computes

```text
Psi_t(Psi_s(z)) - Psi_(s+t)(z).
```

Interpret its norm at the scale of integration tolerance, floating precision, and dynamics.
A small defect does not measure agreement with the Galerkin field or the original PDE.

## Persistence contract

Version-2 `.gns` checkpoints store the unit-ball-normalized network configuration, effective
training radius, time scale, neural state, history, metrics, basis signature, dtype/device
metadata, and reference-package revision. They deliberately do not serialize the weak-form
callable or private numerical Galerkin evaluator. Loading therefore requires the same
`NeuralSemigroupProblem` and operational basis.

Keep schema-1 checkpoints loadable with their original unnormalized semantics; never
reinterpret them as schema 2. Any schema change needs explicit versioning, round-trip tests,
old-checkpoint tests, and migration documentation. Treat the current basis signature as a
compatibility guard, not a cryptographic/content proof of basis identity.

Do not use pickle-based loading modes that weaken the present `weights_only=True` boundary
without a documented security and compatibility reason.

## Scientific evaluation protocol

When an agent builds or analyzes an experiment, separate these objects:

1. A continuous exact/high-fidelity PDE reference, when available.
2. The direct Numerical Galerkin Field trajectory on the fixed basis.
3. The learned GNS trajectory on that same coordinate system.

At minimum report or inspect separately:

- initial projection/reconstruction error;
- Galerkin-versus-continuous-reference error;
- neural-field-versus-Galerkin-field error on independent states;
- neural-trajectory-versus-Galerkin-trajectory error;
- total neural-versus-continuous-reference error;
- time-integration refinement where trajectory error is important.

Do not attribute Galerkin truncation, geometry, quadrature, or RK error to neural learning.
Do not infer field accuracy from a few good trajectories, nor trajectory accuracy from a low
sampled field loss. Training loss and validation loss use normalized field coordinates; label
them accordingly when comparing with physical units.

For structure-bearing examples, add the correct independent observable: heat dissipation,
wave energy, mass/mean conservation, equilibrium residual, boundary trace, or another
problem-derived identity. `defect` is useful only for integration composition. Never invent a
universal energy or claim that the sampled ball is positively invariant.

Negative results are valid results. Do not silently alter sampling, add losses, shorten the
time window, suppress warnings, or compare only favorable endpoints to make an experiment
look successful. Diagnose the field, sampling distribution, Lipschitz budget, Galerkin
baseline, and stiffness explicitly.

## Repository map

- `src/galerkin_neural_semigroup/problem.py`: public problem definition, private `ngfield`
  construction, fixed samples/targets, Adam training, validation selection, loading, and
  metadata.
- `src/galerkin_neural_semigroup/semigroup.py`: trained public object, scaled/physical-time
  fields, flow calls, projection/reconstruction delegation, defect, and checkpoints.
- `src/galerkin_neural_semigroup/_network.py`: private tanh MLP, exact spectral projection,
  unit-ball wrapper, and eval cache.
- `src/galerkin_neural_semigroup/_sampling.py`: the two fixed sampling laws and cached target
  scaling.
- `src/galerkin_neural_semigroup/_integration.py`: private explicit RK4 and Dormand--Prince
  integration.
- `src/galerkin_neural_semigroup/_validation.py`: strict shared input validation.
- `src/galerkin_neural_semigroup/__init__.py`: intentionally minimal public exports.
- `docs/design-contract.md`: authoritative fixed method decisions and limitations.
- `tests/_fixtures.py`: small deterministic `ngfield` problems used across tests.
- `examples/heat.py`: minimal public end-to-end example; it is not evidence for a broad
  empirical claim.

Files prefixed with `_` are private implementation modules. Tests may import their helpers to
verify the method; user examples and downstream applications should not.

## Task playbooks

### Write a user example or notebook

1. State the PDE, state components, weak form, autonomous assumptions, and boundary handling.
2. Build geometry, `Space`, restrictions, and an L2-orthonormal basis with public `ngfield`.
3. Construct only `NeuralSemigroupProblem`; do not expose the Galerkin oracle.
4. Explain `N`, `R`, `tau`, sampling law, architecture, Lipschitz budget, sample count, batch
   size, epochs, learning rate, seed, dtype, and device in short unambiguous terms.
5. Train through `problem.train`; do not redefine library internals in the notebook.
6. Use `semigroup.project`, `solve`, and `reconstruct` for learned evolution.
7. Build a direct Galerkin reference only in evaluation code when the experiment explicitly
   compares methods; label it as a reference, not part of the GNS public workflow.
8. Include independent validation and separate error sources. Save animations explicitly if
   promised and treat warnings at their source.

### Diagnose poor learning or trajectories

Check, in order:

1. The weak form, boundary restrictions, basis orthonormality, and direct Galerkin dynamics.
2. Initial projection and whether relevant states lie within radius `R`.
3. Correct `R`/`tau` scaling by comparing `field`, `velocity`, and private targets in tests.
4. Training and independent validation losses under the named sampling measure.
5. Field error as a function of radius and direction on a new diagnostic sample.
6. Whether the Lipschitz budget is too restrictive for the Galerkin field.
7. Optimization behavior without changing the fixed objective.
8. RK refinement and possible stiffness.
9. Galerkin-versus-PDE error before judging neural performance.

Do not use the training sample as the only diagnostic and do not add an auxiliary loss as a
bug fix. A proposed method variant belongs in an explicit experiment/contract discussion.

### Add or change a feature

1. Identify the governing D-number. If the request changes a fixed choice, revise the design
   proposal first and make the change explicit.
2. Add a failing test for the mathematical invariant, shapes, scaling, invalid inputs,
   reproducibility metadata, and checkpoint behavior as applicable.
3. Implement in the narrow owning module without exposing private mechanisms.
4. Verify both `float64` and relevant `float32`, arbitrary batch axes, CPU, and CUDA when
   available.
5. Update README, design contract, example, changelog, and dependency pin where affected.
6. Run focused and full checks.

### Change the `ngfield` dependency

1. Read the candidate NGF commit's `AGENTS.md`, changelog, public API, and design contract.
2. Install that exact revision locally and run all GNS tests.
3. Verify construction metadata, basis signature, projection/reconstruction shapes,
   quadrature metadata, field batching, dtype/device, and autograd.
4. Update the full commit hash in `pyproject.toml` and the abbreviated recorded revision in
   training metadata together.
5. Rebuild the package to ensure the direct reference is valid in wheel/sdist metadata.

## Testing and verification

For ordinary development with the pinned dependency and Python 3.11 or newer:

```bash
python -m pip install -e ".[dev]"
```

For simultaneous work on sibling checkouts, deliberately use the local NGF source:

```bash
python -m pip install -e "../Numerical-Galerkin-Field[dev]"
python -m pip install -e . --no-deps
```

Use focused tests while iterating:

| Change area | Minimum focused coverage |
| --- | --- |
| Problem/API validation and private reference | `tests/test_problem.py`, `tests/test_space_contract.py` |
| Sampling and target scaling | `tests/test_sampling.py` |
| Architecture, Lipschitz projection, eval cache | `tests/test_network.py` |
| Direct loss, validation selection, checkpoints | `tests/test_training.py` |
| Time scaling, flow, differentiation, defect | `tests/test_flow.py` |

Before declaring a repository change complete, run the CI-equivalent sequence:

```bash
python -m pytest
ruff check .
ruff format --check .
python -m build
```

For changes to user workflow, also execute a reduced deterministic example or a dedicated
small acceptance script. `examples/heat.py` uses a real training budget and need not be run
for every documentation-only change, but any edit to it must at least be syntax-checked and
kept executable.

Do not loosen tolerances or reduce assertions simply because optimization is stochastic.
Tests should use seeds, small linear/analytic fields, structural bounds, and robust error
margins. Do not require a particular GPU for the default suite. When CUDA is available, run
the marked/device-relevant cases without presenting CPU-only success as CUDA verification.

## Implementation and style rules

- Python 3.11+; format and lint with the repository's Ruff configuration.
- Prefer batch-first vectorized PyTorch operations. Avoid per-state loops and repeated
  `.item()`/CPU synchronizations in hot training or integration paths.
- Preserve strict finite, shape, dtype, device, and ambiguity checks. Reject booleans where
  Python would otherwise treat them as integers.
- Do not add dependencies when PyTorch and `ngfield` already provide the needed operation.
- Keep optimizer gradients, target detachment, and eval-state gradients conceptually
  separate.
- Keep metadata serializable with `torch.save(...)/torch.load(..., weights_only=True)`.
- Public docs and code are English; preserve that language unless a user explicitly requests
  a translation artifact.
- Public examples import package-root names. They do not import underscore modules.
- Add an `Unreleased` changelog entry for user-visible behavior. Do not bump versions,
  publish packages, create releases, or push changes unless the user's request authorizes it.

## Current non-goals and prohibited shortcuts

Do not claim or silently add any of the following under the current contract:

- training on trajectories instead of the complete reduced field;
- time as a network input or non-autonomous weak forms;
- extra loss terms, Jacobian/JVP targets, semigroup penalties, or adaptive sampling;
- alternative architectures, activations, optimizers, schedulers, or approximate spectral
  certificates presented as the implemented method;
- user-visible construction or serialization of the numerical Galerkin reference;
- automatic positive invariance of the training ball;
- uniform field-error certificates from finite validation loss;
- convergence from the learned ODE to the original infinite-dimensional PDE;
- implicit/IMEX integration or stiffness guarantees;
- checkpoint basis identity stronger than the data actually checked;
- notebooks that shadow library code with locally redefined implementations.

A request for one of these is a proposed method extension. Explain which invariant changes,
how it would be compared with the existing baseline, and what contract/tests are required.

## Code review rules

Flag any change that:

- exposes the Galerkin reference through the normal public API;
- changes `x=z/R`, target scaling, `field`, `velocity`, or physical-time conversion;
- resamples or adapts the fixed datasets;
- changes the direct loss reduction or adds a hidden objective;
- makes the neural field time-dependent;
- replaces exact spectral projection or breaks the Lipschitz budget;
- recomputes spectral norms during every eval-stage ODE call;
- detaches differentiable input states or retains target/reference graphs;
- loses arbitrary leading batch axes, dtype/device strictness, or finite checks;
- confuses composition defect with model accuracy;
- breaks schema-1 loading or schema-2 round trips;
- advances the NGF pin without full cross-repository tests;
- reports a scientific guarantee not established by the implementation.

## Definition of done

A task is complete only when:

- behavior matches the accepted GNS method and the exact pinned NGF contract;
- focused tests cover scaling, shapes, failure modes, and persistence as relevant;
- the full suite, lint, format check, and build pass without hiding warnings;
- public code uses only the compact public API and does not duplicate internals;
- documentation and changelog describe the implemented behavior precisely;
- scientific comparisons separate Galerkin, neural, temporal, and continuous-reference
  errors;
- the final report states what changed, what was verified, and any limitation not tested.
