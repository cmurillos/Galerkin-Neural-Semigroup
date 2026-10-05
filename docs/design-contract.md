# Design contract

This document states the implemented method. It distinguishes the PDE state
space from the Sobolev regularity of the **reduced vector field as a function of
its coordinates**. The earlier sampling/taper/time-scale variants survive only
when loading historical checkpoints.

## D-001 — Galerkin reference and physical state

An operational real L²-orthonormal basis `(φ₁,…,φ_N)` is fixed by NGF. The PDE
is posed through its autonomous weak form in the underlying L²-based physical
setting. With `Φ(z)=Σ_i z_i φ_i`, NGF evaluates the reduced field

```text
G: ℝ^N → ℝ^N,       G_i(z)=a(Φ(z);φ_i).
```

The precise domain and differentiability of `G` depend on the weak problem and
basis. The L² isometry holds **on the finite-dimensional span** of this basis:
`||Φ(z)||_{L²}=||z||₂`. It does not identify the full L² space with `ℝ^N`.

The normal constructor takes `basis`, `weak`, `radius=R>0`, optional
`quadrature`, and integer `sobolev_order=k≥0`. The compatibility constructor
`from_galerkin` accepts an explicit `GalerkinProblem`. The reference remains
private in either route.

## D-002 — Normalized open ball and exact scaling

Let `B_N(R)={z∈ℝ^N:||z||₂<R}` and `x=z/R∈B_N(1)`. Define

```text
g_R(x)=G(Rx)/R,       Fθ(z)=R fθ(z/R),       z∈B_N(R).
∂_x^α g_R(x)=R^(|α|-1) ∂_z^α G(Rx).
```

This is an exact change of coordinates, with **unchanged physical time**.
Changing `R` generally changes `g_R` and calls for new training. Additional
radius transfer is justified only for a separately established symmetry of a
particular problem. There is no arbitrary time normalization parameter.

The model's ODE field is restricted to the open ball. Direct `field` and
`velocity` calls at or beyond its boundary raise. The separate
`velocity_zero_extended` convention equals the interior field for `||z||<R`
and exactly zero for `||z||>=R`; it is never used to integrate or continue a
trajectory. This convention may jump and does not inherit the interior
Lipschitz bound. A local solve stops at the first numerical boundary contact
and reports `DomainExitError`. No invariance or global flow is asserted.

## D-003 — Indexed H^k objective and sample measure

NGF lists all multi-indices `α∈ℕ₀^N` with `|α|≤k`, **once each**, and computes
`∂_z^αG` with state-coordinate automatic differentiation. The continuous target
quantity is the standard unweighted integer Sobolev norm:

```text
||fθ-g_R||²_{H^k(B_N(1))}
= ∫_{B_N(1)} Σ_{|α|≤k} ||∂_x^α fθ(x)-∂_x^α g_R(x)||₂² dx.
```

For `B` independent fixed samples from uniform volume on the unit ball,
training minimizes

```text
L_B(θ)=(1/B) Σ_{j=1}^B Σ_{|α|≤k}
         ||∂_x^α fθ(x_j)-R^(|α|-1)∂_z^αG(Rx_j)||₂².
```

The sums commute. The empirical average estimates the integral divided by
`|B_N(1)|`; this constant does not change the minimizer. There are no
factorials, dimension-dependent normalization of coordinate components,
extra loss terms, or trajectory supervision. For `k=0` the objective reduces
to field matching. `H^k` measures smoothness in finite-dimensional state
coordinates; it does not promote the physical PDE state to a spatial H^k space.

Sampling uses `x=s^(1/N)ξ/||ξ||₂` with `ξ` standard Gaussian and
`s∼Uniform(0,1)`. Train and validation samples are drawn once and held fixed;
the latter is independent. Samples that round onto the boundary in unit or
physical coordinates are regenerated; every training point is strictly interior. Targets are prepared in batches, detached **after**
differentiation, cached and checked for finiteness. Gradients through the neural
field derivatives remain available for Adam.

A target `H^k` formulation requires `G` and `fθ` to possess the corresponding
weak derivatives on the ball and square-integrability of their differences.
Finite sampled derivatives do not by themselves prove those analytic hypotheses.
For higher `k`, the chosen `tanh` network is smooth, but weak-form evaluation
and numerical differentiation must still be suitable for the problem.

## D-004 — Free weights and a posteriori Lipschitz bound

`fθ:B_N(1)→ℝ^N` is the restriction of an affine `tanh` MLP with free weights.
There is no spectral projection, preassigned budget, automatic calibration
or penalty on its layer norms. For each fixed finite set of weights,
`Lθ=product_l ||W_l||₂` bounds its Lipschitz quotient because `tanh` is
1-Lipschitz. This also bounds the physical field inside `B_N(R)`, by exact
coordinate scaling. It need not be uniform across networks or dimensions.
The measured layer norms and their product are diagnostics, not training
constraints or trajectory-accuracy certificates. Eval weights are detached
from parameters; input-state derivatives remain available.

## D-005 — Integration and domain

NGF owns `integrate_field`. The Galerkin and neural solves call this single
implementation. Set `J₁[X]=X`, `J_{r+1}[X]=DJ_r[X] X`; its degree-`p`
Taylor step is

```text
z_next=z+Σ_{r=1}^p h^r J_r[X](z)/r!.
```

`order=p≥1` is independent of `k`. A positive `step` fixes a maximum internal
step; omitting it selects adaptive stepping with an estimate from the next
Taylor term. `tolerance` controls the adaptive rule and cannot be combined
with `step`. The same `times`, `order`, and time-step controls must be used for
matched Galerkin/neural comparisons. The `radius=R` check can also be passed
to `G.solve` to restrict both comparisons to the same domain. The algorithm
evaluates the field only at accepted interior states; a candidate outside is
not returned. First contact is localized using roots of
`||P(u)||₂²-R²` for the accepted degree-p Taylor polynomial `P`, including
contacts followed by reentry. Only jets at the interior start are evaluated.
The event time/state and contacting batch mask are separate from completed
interior outputs and the last accepted interior state. In adaptive mode the
error check precedes event detection, so rejected polynomials do not declare
an exit. Event data approximate the ODE contact and depend on integration
accuracy; they do not certify the exact exit or prolong the solution.

The adaptive estimate does not certify global accuracy. Taylor differentiation
can be expensive at high order and explicit time stepping can be problematic
for stiff fields. The exact local flow has identity and composition where the
solutions exist; `defect` measures only numerical composition error.

## D-006 — Training and persistence

Adam, uniform volume, fixed validation, free affine weights and cached
reference targets are fixed choices. Training exposes architecture widths,
dataset size, minibatch size, epochs,
learning rate, seed, device and dtype. The best independent validation epoch
is restored. Metadata records `R`, `k`, normalization, training settings,
quadrature details, exterior/exit convention and exact NGF source revision.

Schema-6 checkpoints record free weights, the open-ball domain, the exterior-zero
convention and `k` with model weights.
Loading checks reduced dimension, basis signature, radius, domain and Sobolev
order. The basis signature is a compatibility guard, not a content hash.
Older schemas 1–5 retain spectral projection; schemas 1–4 also retain their own
field normalization, time scaling and historical support rules. They are never
silently converted to free-weight models. New training
does not generate those schemas.

## D-007 — Scope of a comparison

Separate physical projection/reconstruction error, Galerkin truncation error,
field approximation error, temporal integration error and PDE-reference error.
Use independent field samples and matched integration settings on the interval
where both trajectories exist. A small empirical H^k loss alone gives no
uniform field certificate or trajectory bound. Even a good Galerkin-field
surrogate does not establish convergence of the Galerkin approximation to the
underlying PDE.

## D-008 — Independent evaluation on the normalized ball

The new local model accepts explicitly reserved physical-coordinate states in
`evaluate_field(states)` and initial states and strictly monotone times in
`evaluate_trajectories(initial_states, times)`. All returned state, field,
trajectory, integral and rate quantities use `x=z/R` and `g_R=G(Rx)/R`; time is
unchanged. Evaluation neither trains nor changes saved training metrics. It
does not sample outside the open ball, infer an invariant, or claim independent
data if the caller reuses training or validation states.

Field evaluation reports RMS value error, RMS indexed derivative errors for
each `0 <= r <= k`, their H^k sum, a profile in normalized radial intervals,
and maximum sampled operator norms of first derivatives for both fields.
NGF supplies the reference indexed derivatives; GNS evaluates the learned
ones. The maximum sampled norm of `Dg_R` is an empirical lower estimate of its
supremum, not a certified global Lipschitz upper bound. The existing network
product-of-layer-norms bound is reported separately. Empty radial bins have zero count and
an undefined (`nan`) RMS value.

Trajectory evaluation uses NGF's same Taylor integrator, order, times and
step/tolerance settings on both fields. It reports normalized trajectory
errors, same-state field and radial-rate gaps along learned paths, the
normalized `L²` norm and its rate `x·X(x)`, and per-component integrals using
`G.integral_weights()`. Integral and norm changes from the initial state are
observations; the generic weak problem need not conserve either quantity.
Both integrations stop at the open-ball boundary. Each output-time aggregate
is conditioned on trajectories for which the required states still exist;
the counts and separate survival fractions accompany these values. Missing
states are `nan`, and a missing exit time means no exit was observed by the
last requested time. The recorded exit time and boundary state describe the numerical contact;
last accepted interior times are separate report keys. `comparison_exit_time`
is the first contact of either field in the integration direction (the earlier
time forwards, the later time backwards). Error and field/rate-gap aggregates
use only common survivors. Boundary events are not appended to the paths. Optional time-step refinement is a
numerical indicator, not a certified error bound; a refined trajectory may
exit before its coarse counterpart.

## D-009 — Function-valued study and indexed derivative option

`Geometry`, `Space`, restrictions and weak-form operators exposed by GNS
refer to the exact NGF objects. `System(basis,weak,radius,sobolev_order,...)`
defines a study, `study.train(...)` produces a function-valued `Model`, and
`study.load(path)` reconstitutes a matching local model. The historical
`NeuralSemigroupProblem` and `NeuralSemigroup` remain available unchanged.
The new workflow requires an open-ball schema-5 or schema-6 model; it does not reinterpret
the historical time and taper semantics of schemas 1–4.

`model.state(u0)` projects a physical initial function, while
`model.from_coefficients(z)` explicitly accepts coordinates `[...,N]` strictly
inside the model's ball. These NGF `State`, `Function` and `Solution` types
perform the same spatial evaluation, componentwise integral and L² norm as
the numerical workflow. `model.evolve(initial,times,order,step|tolerance)`
uses the shared Taylor integrator and returns only completed requested times
if a numerical boundary exit occurs. `solution.exit_status()` then also
contains the last accepted interior state/time and a separate numerical
boundary event. The event is a `Function`, not an admissible phase `State`. `solution.at(t)` accepts
only recorded output times; it does not interpolate.

For `z=state.coefficients()`, `state.velocity()` represents the physical
learned velocity `Fθ(z)`. `state.indexed_derivatives(k)` maps every
`|alpha|<=k` to the function with coefficients `∂_z^alpha Fθ(z)`. These
derivatives are not normalized-unit-ball loss derivatives: for the loss
coordinates `x=z/R`, the exact conversion follows D-002. A derivative vector
is a function in `V_N`, not necessarily a phase state inside the ball. The
temporal Taylor order remains independent of `k`.

`model.metrics` delegates projection and reference quadrature indicators to
NGF, and its time refinement and rates to the learned field in physical
coordinates; sampled Lipschitz is a finite sample maximum. The separate
`model.evaluate.field(...)` and `.trajectories(...)` return structured report
objects with methods for every D-008 family and `raw()` for all report keys.
Evaluation is normalized on the unit ball. These report objects do not train
or alter checkpoint metrics. Model training history and checkpoint metrics
remain available separately.

## D-010 — Internal efficiency and bilateral time evolution

The public calls from D-009 remain the same. The loss computes one average
after summing squared errors over every output component and multi-index;
this is the same empirical norm of D-002 without derivative weights. The
reference targets remain cached on fixed sampled states. If the GPU cannot
comfortably hold them, they are stored on CPU and each selected batch is
transferred without changing its values or sampling order. Affine weights are reused within a batch without projection;
no cached training graph survives an optimizer step.

`Model` initializes `ngfield.FunctionalFlow` with a compatible autonomous
field. Independent evaluation accepts strictly increasing or decreasing
time grids, including negative values, with the initial state at the first
requested time. Prefixes of paths that leave the ball come from NGF's
already accepted output states. Reports retain the requested order and their
normalized metrics; numerical exit times remain approximate.

## D-011 — Local flow until first boundary contact

For each initial point inside the open ball, the model represents the unique
maximal interior ODE solution. Identity, composition and inverse relate only
admissible domains: `Psi_t:D_t→D_-t`, with inverse `Psi_-t`. These domains
may be proper subsets or empty; the model is not a group on the entire span.
Comparison with Galerkin ends at the first exit of either trajectory. A
boundary event terminates evaluation, without clipping, freezing, restarting
or integrating the exterior-zero convention. Fixed-radius exits neither
prove PDE blowup nor indicate numerical solver failure. New schema-6 models
use the free-weight contract of D-004; loading schemas 1–5 preserves their
historical field meanings.
