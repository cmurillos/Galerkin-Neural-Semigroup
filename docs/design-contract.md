# Design contract

This document records the initial public contract of Galerkin Neural Semigroup. It
separates choices fixed by the method from information that a user must supply.

## D-001 — Public problem definition

The normal constructor is

```python
NeuralSemigroupProblem(
    basis=basis,
    weak=weak,
    radius=R0,
    quadrature=None,
    time_scale=1.0,
)
```

The operational basis is fixed, real and numerically L2-orthonormal. In the modern
Numerical Galerkin Field workflow it carries its `Space`, geometry, physical component
shape and homogeneous restrictions. These data are not repeated here. `basis.dimension`
is the reduced dimension `N`.

`weak` is a complete autonomous weak form in the Numerical Galerkin Field expression
language. `radius` is the positive radius of the coordinate ball used for learning.
`quadrature` retains the exact semantics of the reference package: `None` is automatic,
an integer fixes an order and a real in `(0,1)` requests adaptive preparation.

The compatibility constructor `from_galerkin(problem, ...)` is reserved for explicit
`GalerkinProblem` workflows. Neither route returns the numerical reference field.

## D-002 — Initial measure and component-balanced Sobolev objective

Initial training states are independent samples from normalized Lebesgue volume on

```text
B_N(R0) = {z in R^N : ||z||_2 < R0}.
```

For a standard Gaussian direction `xi` and `s ~ Uniform(0,1)`, the implementation uses

```text
z = R0 * s^(1/N) * xi / ||xi||_2.
```

Let `Y = tau * G(Z)` be the cached scaled Galerkin target. Component scales are computed
once from the initial training design and remain fixed after adaptive refinements:

```text
s_j^2 = mean_i Y_j(Z_i)^2,

W = diag((s_j^2 + epsilon * mean_k s_k^2)^(-1/2)).
```

If the reference field is identically zero, `W` is the identity. Thus `epsilon > 0` is
a relative floor rather than a dimensional constant, and global changes of target scale
do not alter the balance between components.

Each state receives one Rademacher vector `xi` whose coordinates are independently
`-1` or `1`. The stored Jacobian direction is `V = R0 * xi`; multiplication by `R0`
expresses the derivative in unit-ball coordinates without changing the public state
coordinates. Numerical Galerkin Field supplies the exact directional derivative
`J_Y(Z)V = tau * J_G(Z)V` through forward-mode automatic differentiation.

For `lambda_J >= 0`, the per-state terms are

```text
L_value = mean_j [W(F(Z)-Y)]_j^2,

L_jacobian = mean_j [W(J_F(Z)V-J_Y(Z)V)]_j^2,

L = L_value + lambda_J * L_jacobian.
```

The inverse-RMS matrix prevents large target components from determining the entire fit.
The JVP term supervises the first-order variation of the field without materializing a
full Jacobian at every state. Values and JVP targets are evaluated in batches, detached
from autograd and cached. Validation uses independent states and directions with the same
fixed component weights.

Adaptive refinement, when requested, does not replace this objective. It changes the
empirical design by appending states in regions where the current field has a persistent
coverage error. Its selection score is the same per-state Sobolev loss:

```text
eta(z) = L_value(z) + lambda_J * L_jacobian(z).
```

## D-003 — Fixed neural architecture

The learned field is autonomous and has signature

```text
F: [...,N] -> [...,N].
```

It is an affine MLP with componentwise `tanh` between affine layers and no activation
after the output layer. Time is never an input. `hidden=()` selects one affine layer and
is the exact linear test architecture.

Every raw weight `W_k` is replaced during evaluation by

```text
Wbar_k = W_k / max(1, ||W_k||_2 / gamma).
```

If there are `d` affine layers and the user requests global budget `Lmax`, the package
sets `gamma = Lmax^(1/d)`. Hence `Lip(F) <= Lmax` independently of depth. Spectral norms
are computed by `torch.linalg.matrix_norm(..., ord=2)`; a finite power-iteration estimate
is not used as a certificate.

During optimization the projection remains in the differentiable forward map. Once
the module enters evaluation mode, the projected weights are cached because they are
fixed; ODE integration therefore does not recompute matrix norms at every stage.
The cache is detached from the parameters but not from input states: state Jacobians
remain available. Calling `semigroup.field.train()` invalidates the cache and restores
parameter differentiation when it is explicitly required.

## D-004 — Time scaling

With `time_scale=tau`, targets approximate `tau * G(z)` and the trained field evolves in
scaled time `s=t/tau`. Public `solve` accepts physical times and performs this conversion
internally. Thus `semigroup.field(z)` is the scaled field and
`semigroup.velocity(z) = semigroup.field(z)/tau` is the learned physical-time velocity.

## D-005 — Training interface

The public training call receives only choices not fixed by the method:

```python
problem.train(
    hidden=...,
    lipschitz=...,
    samples=...,
    batch_size=...,
    epochs=...,
    max_epochs=None,
    tolerance=None,
    max_time=None,
    refine_every=None,
    refine_samples=512,
    candidate_samples=32768,
    patience=100,
    lr=...,
    jacobian_weight=0.1,
    balance_epsilon=1e-6,
    seed=...,
)
```

Adam, one radius-scaled Rademacher direction per state, validation sampling and cached
targets are currently fixed. `jacobian_weight` is nonnegative and `balance_epsilon` is
strictly positive. Device selection defaults to CUDA when available and otherwise CPU.
Computation uses float64 by default; `device` and `dtype` are explicit advanced overrides
because they affect reproducibility.

`epochs` is the minimum budget whenever `max_epochs` is larger; omitting `max_epochs`
recovers the previous exact epoch budget. `max_time` is a hard wall-clock budget in
seconds, checked after each epoch. After the minimum budget, `tolerance` accepts the fit
when the 99th percentile of the independent refinement score is below the requested
level. Two unsuccessful plateau checks terminate with `stop_reason="stalled"`.

Setting `refine_every` enables refinement checks at that epoch interval. A check is
eligible only after `patience` epochs without a relative validation improvement of
`1e-3`. The candidate design uses randomized directions and at most 32 radial strata;
it is a search probe, not the final training measure. Let `eta_i` be candidate scores.
To prevent a single numerical outlier from dominating the design, scores are capped at
their 99th percentile. Local kernel centers are sampled with probabilities proportional
to

```text
(min(eta_i, quantile_0.99(eta)) + machine_epsilon)^1.5.
```

Kernel bandwidth is determined by the eighth-neighbor distance within the 512
candidates of largest weight and clipped between `0.005 R0` and `0.15 R0`. Gaussian
proposals outside the ball are rejected. The refinement mixture uses 85% local
proposals and 15% global probe states. Refinement occurs only when the candidate 99th
percentile is at least 25% larger than its training counterpart; this separates a
coverage deficit from an optimization or capacity deficit. For half of the following
adaptation window, half of each epoch design is drawn from the newly appended states.

Without an independent probe, the returned weights minimize validation loss. With
tolerance or adaptive refinement enabled, they minimize the probe's 99th-percentile
score in the latest adaptive stage. All epochs are retained in `semigroup.history`.
Total, value and Jacobian histories, candidate checks and refinement events are reported
separately, together with the stop reason, elapsed time, final sample count, component
scales, layer spectral norms and their product.

## D-006 — Flow and numerical integration

`NeuralSemigroup` represents the exact continuous flow mathematically, while `solve`
provides a numerical realization. Omitting `step` uses explicit Dormand--Prince 5(4);
providing a positive `step` uses fixed-step RK4. The two controls cannot be combined.

`defect(z,s,t)` evaluates the numerical quantity

```text
Psi_t(Psi_s(z)) - Psi_(s+t)(z).
```

It diagnoses the integrator and floating-point arithmetic, not a learned semigroup-loss
term.

## D-007 — Persistence and provenance

Checkpoints contain neural parameters, architecture, `R0`, time scale, optimization
history, basis signature, precision, device and the pinned reference-package revision.
They do not serialize the weak-form evaluator or the private numerical field. Loading
therefore occurs through the same `NeuralSemigroupProblem` and rejects incompatible
dimensions, signatures, radii or time scales.

The current basis signature is deliberately conservative: dimension, physical value
shape, family and component allocation when available. The user remains responsible for
supplying the same ordered operational basis; a stronger content hash is future work.

## D-008 — Scope

- Training controls a field only on the sampled reduced ball; the ball is not asserted
  to be positively invariant.
- A small empirical loss or probe quantile is not a certified uniform error.
- Adaptive refinement can only discover difficult regions represented by its finite
  radially stratified probe.
- Exact spectral projection guarantees global well-posedness of the learned ODE but does
  not establish convergence to the original infinite-dimensional evolution.
- Explicit integration may be expensive for stiff reduced dynamics.
- The fixed MLP is the method implemented here. Alternative neural architectures are not
  part of the initial public API.
