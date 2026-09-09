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

## D-002 — Fixed learning measure and relative-angular objective

Training states are independent samples from normalized Lebesgue volume on

```text
B_N(R0) = {z in R^N : ||z||_2 < R0}.
```

For a standard Gaussian direction `xi` and `s ~ Uniform(0,1)`, the implementation uses

```text
z = R0 * s^(1/N) * xi / ||xi||_2.
```

Let `Y = tau * G(Z)` be the cached scaled Galerkin target, let `lambda_ang >= 0` and let
`epsilon > 0`. For each sampled state the two loss components are

```text
L_rel = ||F(Z)-Y||_2^2 / (||Y||_2^2 + epsilon),

L_ang = 1 - <F(Z),Y>
              / sqrt((||F(Z)||_2^2 + epsilon)(||Y||_2^2 + epsilon)).
```

The empirical objective is the batch mean of

```text
L = L_rel + lambda_ang * L_ang.
```

The relative term prevents large target velocities from determining the entire fit,
while the angular term explicitly distinguishes aligned, orthogonal and opposing vector
fields. `epsilon` keeps both terms finite near stationary states. Cosine values are
clamped to `[-1,1]` only to remove floating-point excursions; this does not change the
formula in exact arithmetic. Targets are evaluated in batches, detached from autograd
and cached. Validation uses an independent sample and the same loss parameters.

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
    lr=...,
    angular_weight=0.1,
    loss_epsilon=1e-8,
    seed=...,
)
```

Adam, the form of the relative-angular loss, validation sampling and cached targets are
currently fixed. `angular_weight` is nonnegative and `loss_epsilon` is strictly positive.
Device selection defaults to CUDA when available and otherwise CPU. Computation uses
float64 by default; `device` and `dtype` are explicit advanced overrides because they
affect reproducibility.

The returned weights are those with minimum independent validation loss. All epochs are
retained in `semigroup.history`. Total, relative and angular histories and final metrics
are reported separately, together with the layer spectral norms and their product.

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
- A small empirical loss is not a certified uniform error.
- Exact spectral projection guarantees global well-posedness of the learned ODE but does
  not establish convergence to the original infinite-dimensional evolution.
- Explicit integration may be expensive for stiff reduced dynamics.
- The fixed MLP is the method implemented here. Alternative neural architectures are not
  part of the initial public API.
