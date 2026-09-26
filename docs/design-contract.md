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

## D-002 — Three fixed learning measures and direct objective

The physical training domain is

```text
B_N(R0) = {z in R^N : ||z||_2 < R0}.
```

The network sees normalized coordinates `x = z/R0` in `B_N(1)`. For a standard
Gaussian direction `xi` and `s ~ Uniform(0,1)`, the implementation offers three
fixed, non-adaptive measures:

```text
sampling="volume": x = s^(1/N) * xi / ||xi||_2,
sampling="radius": x = s       * xi / ||xi||_2,
sampling="mixed":  choose "volume" or "radius" independently with probability 1/2,
physical state:    z = R0 * x.
```

The first is normalized Lebesgue volume on the ball. The second is the product of the
uniform radial measure on `(0,R0)` and uniform angular measure on the unit sphere.
The third is their equal-probability mixture, with a new Bernoulli choice for every
sample. It is not uniform volume or an adaptive sampling law.
Directions, radii, normalized states and physical target states are generated
vectorially. Training and validation states are drawn once from independent samples of
the same selected measure. Epochs shuffle the fixed training tensor; they neither
resample it nor change its distribution.

For the normalized target

```text
G_hat(x) = (tau/R0) * G(R0*x),
```

the sole empirical objective is the batch mean of the squared Euclidean field
difference:

```text
L_B(theta) = ||F_hat_theta(X) - G_hat(X)||_F^2 / B.
```

It sums rather than averages over the coordinate dimension. Galerkin targets are
evaluated in vectorized batches, detached from autograd and cached. There is no relative,
angular, derivative, Jacobian or adaptive-refinement term.

## D-003 — Fixed neural architecture

The normalized neural field is autonomous and has signature

```text
F_hat: [...,N] -> [...,N].
```

It is an affine MLP with componentwise `tanh` between affine layers and no activation
after the output layer. Time is never an input. `hidden=()` selects one affine layer and
is the exact linear test architecture.

Every raw weight `W_k` is replaced during evaluation by

```text
Wbar_k = W_k / max(1, ||W_k||_2 / gamma).
```

If there are `d` affine layers and the user requests global budget `Lmax`, the package
sets `gamma = Lmax^(1/d)`. The field exposed in physical reduced coordinates is

```text
F_base(z) = R0 * F_hat(z/R0).
```

The input and output factors cancel in its Lipschitz quotient, so
`Lip(F_base) = Lip(F_hat) <= Lmax` independently of depth. The deployed field now
also receives the fixed compact-support taper of D-009, whose global Lipschitz
bound can exceed the requested base-network budget. Spectral norms are computed by
`torch.linalg.matrix_norm(..., ord=2)`; a finite power-iteration estimate is not used as
a certificate.

During optimization the projection remains in the differentiable forward map. Once
the module enters evaluation mode, the projected weights and fixed biases are cached
and detached from the parameters; ODE integration therefore neither recomputes matrix
norms at every stage nor builds a parameter graph for ordinary states. The input is not
detached: state Jacobians remain available whenever the input explicitly requires a
gradient. Calling `semigroup.field.train()` invalidates the cache and restores parameter
differentiation when it is explicitly required.

## D-004 — Time scaling

With `time_scale=tau`, the normalized targets approximate
`(tau/R0) * G(R0*x)`. After the inverse coordinate transform, the public field
approximates `tau * G(z)` and evolves in scaled time `s=t/tau`. Public `solve` accepts
physical times and performs this conversion internally. Thus `semigroup.field(z)` is the
scaled field in the original reduced coordinates and
`semigroup.velocity(z) = semigroup.field(z)/tau` is the learned physical-time velocity.

## D-005 — Training interface

The public training call receives only choices not fixed by the method:

```python
problem.train(
    hidden=...,
    lipschitz="auto",  # or a positive manually specified spectral budget
    lipschitz_factor=1.5,
    samples=...,
    sampling="volume",  # or "radius" or "mixed"
    batch_size=...,
    epochs=...,
    lr=...,
    seed=...,
)
```

Adam, the direct field loss and cached targets are fixed. `sampling` selects one of the
three measures in D-002 and does not change during training. Device selection defaults to
CUDA when available and otherwise CPU. Computation uses float64 by default; `device`
and `dtype` are explicit advanced overrides because they affect reproducibility.
The optional calibration of the base network budget is specified in D-010;
it does not add a training objective or a certificate of the reference field.

The returned weights are those with minimum independent validation loss. Training and
validation losses are values of the normalized direct objective in D-002. All epochs are
retained in `semigroup.history`; final metrics include the layer spectral norms and their
product.

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

Version-4 checkpoints contain neural parameters, architecture, the unit-ball
normalization marker, `R0`, time scale, optimization history, basis signature, precision,
device, support start/end radii, empirical calibration metadata and the pinned
reference-package revision. They do not serialize the weak-form
evaluator or the private numerical field. Loading therefore occurs through the same
`NeuralSemigroupProblem` and rejects incompatible dimensions, signatures, radii or time
scales. Version-1 and version-2 checkpoints remain loadable with their original
unnormalized and unmasked field semantics; version-3 retains its former taper
on radii 0.9R0 to R0. None is silently reinterpreted.

The current basis signature is deliberately conservative: dimension, physical value
shape, family and component allocation when available. The user remains responsible for
supplying the same ordered operational basis; a stronger content hash is future work.

## D-008 — Scope

- Training controls the raw field only through a finite sample from the selected measure.
  The raw neural field matches the deployed field throughout the training ball;
  no agreement with Galerkin is promised in the exterior taper. The continuous
  learned flow can leave the training ball, but cannot cross the stationary
  support boundary at twice its radius.
- A small empirical loss is not a certified uniform error.
- Exact spectral projection guarantees global well-posedness of the learned ODE but does
  not establish convergence to the original infinite-dimensional evolution.
- Explicit integration may be expensive for stiff reduced dynamics.
- The fixed MLP is the method implemented here. Alternative neural architectures are not
  part of the initial public API.

## D-009 — Compact support after training

Newly trained models use normalized coordinates `x=z/R0`, `r=||x||_2` and
the nonexpansive radial projection `P(x)=x/max(1,r)`. The raw field is fitted
to the unmodified Galerkin target throughout the sampled ball. After training:

```text
F_supported(z) = chi(r) * R0 * F_hat(P(x)),
chi(r) = 1                         for r <= 1,
       = 1 - 3q^2 + 2q^3           for 1 < r < 2, q=r-1,
       = 0                         for r >= 2.
```

The same multiplier applies to `field`, physical-time `velocity`, and the integrated
flow. For `1<r<2` the core is evaluated at the projected boundary point;
beyond `2` it is not evaluated. The resulting field is continuous and globally
Lipschitz, including at both transition radii. No additional training
objective or learned parameter is added. The output equals the raw field
throughout the training ball and is generally biased relative to Galerkin
outside it. The historical `supported_validation_loss` metric equals the
deployed field's validation loss inside the sampled ball.

If `L_base` bounds the raw field's Lipschitz quotient and `a=||F_hat(0)||_2`, a valid
global bound for the deployed field is `L_base + 1.5*(a + L_base)`;
the radius normalization cancels. This **can exceed** the user-specified
base budget; moving the transition outward does not eliminate its contribution.
The continuous flow is globally unique. Numerical trajectories can have
small integration error; the algorithm does not silently clip states.
Only states starting at or outside `2R0` are stationary.

## D-010 — Empirical automatic spectral budget

For normalized reference `G_hat(x)=(tau/R0)G(R0*x)`, the ideal base-field
Lipschitz constant on `B_N(1)` is the supremum of its two-point quotients.
When `lipschitz="auto"` (the default), use independent fixed anchors sampled
from the mixed ball and the origin; evaluate symmetric coordinate difference
quotients through the private Galerkin field, assemble an approximate Jacobian
and take the maximum operator 2-norm across anchors. All pairs remain inside
the unit ball and use batched reference evaluations. Multiply the estimate
by `lipschitz_factor` (default 1.5) and use at least 1e-6 as the positive base
network spectral budget. Record the estimate, margin, probe count, finite
difference step, selected budget and `certified_upper_bound=False`.

Finite probes and finite difference steps do not provide a global upper
bound for a general nonlinear reference; the margin is heuristic. A manually
supplied positive number bypasses reference calibration. For affine
references the quotient equals the operator norm of the linear part;
the same estimator recovers it up to floating-point error.
