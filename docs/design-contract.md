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

## D-002 — Radial design and component-balanced Sobolev objective

For `m` initial states in dimension `N`, the method fixes

```text
L = min(m, max(2, min(32, round(sqrt(m/N)))))

r_l = l R0/(L-1),  l = 0, ..., L-1.
```

It retains `z=0` once. The other states are distributed as evenly as possible over the
positive radii. At each such radius it draws an independent standard Gaussian `xi`,
normalizes it, and sets

```text
u = xi/||xi||_2,    z = r_l u.
```

Thus angular samples are uniform on the unit sphere, the origin and boundary are both
represented, and the radial design does not acquire the high-dimensional concentration
near `R0` of normalized Lebesgue volume. Independent validation uses new directions on
the midpoint shell of every initial adjacent-layer interval.

Let `Y = tau * G(Z)` be the cached scaled Galerkin target. Component scales are computed
once from the initial training design and remain fixed after adaptive refinements:

```text
s_j^2 = mean_i Y_j(Z_i)^2,

W = diag((s_j^2 + epsilon * mean_k s_k^2)^(-1/2)).
```

If the reference field is identically zero, `W` is the identity. Thus `epsilon > 0` is
a relative floor rather than a dimensional constant, and global changes of target scale
do not alter the balance between components.

Every state receives the complete canonical basis `e_1, ..., e_N`. The directions
`V_k = R0 * e_k` express derivatives in unit-ball coordinates without changing the
public state coordinates. Numerical Galerkin Field supplies all columns
`J_Y(Z)V_k = tau * J_G(Z)V_k` through forward-mode automatic differentiation. Thus
`K=N` at every training state and no random projection of the Jacobian remains.

For `lambda_J >= 0`, the per-state terms are

```text
L_value = mean_j [W(F(Z)-Y)]_j^2,

L_jacobian = mean_(j,k) [W(J_F(Z)-J_Y(Z)) R0]_(j,k)^2,

L = L_value + lambda_J * L_jacobian.
```

The inverse-RMS matrix prevents large target components from determining the entire fit.
The derivative term supervises the complete first-order variation at every training
state. Values and full Jacobian targets are evaluated in batches, detached from autograd
and cached. Independent validation values use the same fixed component weights and are
the sole checkpoint-selection, plateau and stopping objective. A complete-Jacobian audit
on a small validation subset is computed periodically for diagnosis but never selects
the returned model.

Adaptive refinement, when requested, does not replace this objective. It changes the
empirical design by appending states in regions where the current field has a persistent
coverage error. Its independent selection score is the per-state value loss:

```text
eta(z) = L_value(z).
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

Adam, the complete radius-scaled canonical basis, validation sampling and cached targets
are currently fixed. `jacobian_weight` is nonnegative and `balance_epsilon` is strictly
positive. Device selection defaults to CUDA when available and otherwise CPU. Computation
uses float64 by default; `device` and `dtype` are explicit advanced overrides because they
affect reproducibility.

`epochs` is the minimum budget whenever `max_epochs` is larger; omitting `max_epochs`
recovers the previous exact epoch budget. `max_time` is a hard wall-clock budget in
seconds, checked after each epoch. After the minimum budget, `tolerance` accepts the fit
when the 99th percentile of the independent refinement score is below the requested
level. Two unsuccessful plateau checks terminate with `stop_reason="stalled"`.

Setting `refine_every` enables refinement checks at that epoch interval. A check is
eligible only after `patience` epochs without a relative validation improvement of
`1e-3`. The independent candidate design uses new unit-sphere directions distributed
over the midpoint shell of every current adjacent-layer interval. If `I_l` indexes
candidates on `rho_l = (r_l+r_(l+1))/2`, its radial error profile is

```text
E_l = mean_(i in I_l) eta(z_i),
```

The method chooses an index in `argmax_l E_l` and inserts `refine_samples` new states on
the spherical layer of radius

```text
r_new = (r_l + r_(l+1))/2.
```

Choosing the largest, rather than smallest, interval error is essential: the added
layer must increase resolution where the learned field is least accurate between
existing layers. Refinement still occurs only when the candidate 99th percentile is at
least 25% larger than its training counterpart; this separates a coverage deficit from
an optimization or
capacity deficit. The independent probe is rebuilt over the enlarged ordered layer set.
For half of the following adaptation window, half of each epoch design is drawn from the
newly appended states. Candidate history records `(radius, E_l)` for every check, and a
refinement event records the selected interval and its midpoint.

Without an independent probe, the returned weights minimize validation value loss. With
tolerance or adaptive refinement enabled, they minimize the probe's 99th-percentile
value score in the latest adaptive stage. All epochs are retained in
`semigroup.history`. Total and component training histories, value-validation history,
periodic Jacobian audits, candidate checks and refinement events are reported separately,
together with the stop reason, elapsed time, final sample count, component scales, layer
spectral norms and their product.

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

- Training controls a field only on a finite radial design inside the reduced ball; the
  ball is not asserted to be positively invariant.
- A small empirical loss or probe quantile is not a certified uniform error.
- Adaptive refinement resolves radial variation but can only discover angular errors
  represented by its finite unit-sphere probe.
- Exact spectral projection guarantees global well-posedness of the learned ODE but does
  not establish convergence to the original infinite-dimensional evolution.
- Explicit integration may be expensive for stiff reduced dynamics.
- The fixed MLP is the method implemented here. Alternative neural architectures are not
  part of the initial public API.
