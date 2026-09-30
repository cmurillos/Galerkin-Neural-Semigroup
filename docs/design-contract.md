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

The network and its physical field are defined only in the open ball. Direct
field evaluation at or beyond its boundary raises; no taper or projected
exterior values are fitted. Some trajectories leave this ball. In that case a
local numerical solve reports `DomainExitError` instead of returning a value
outside the model's domain. No invariance of the ball is asserted.

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
the latter is independent. Targets are prepared in batches, detached **after**
differentiation, cached and checked for finiteness. Gradients through the neural
field derivatives remain available for Adam.

A target `H^k` formulation requires `G` and `fθ` to possess the corresponding
weak derivatives on the ball and square-integrability of their differences.
Finite sampled derivatives do not by themselves prove those analytic hypotheses.
For higher `k`, the chosen `tanh` network is smooth, but weak-form evaluation
and numerical differentiation must still be suitable for the problem.

## D-004 — Network and empirical spectral budget

`fθ:B_N(1)→ℝ^N` is an autonomous affine `tanh` MLP. The exact spectral
projection applies `W↦W/max(1,||W||₂/γ)` to each affine weight. With `d` layers
and requested budget `L`, `γ=L^(1/d)`; the product of effective layer norms
bounds the core's Lipschitz quotient. The physical coordinate scaling preserves
that quotient within the ball. In eval mode weights are cached and detached
from parameters while derivatives with respect to input states remain available.

A manual positive `lipschitz` sets `L`. By default GNS samples the indexed first
state derivatives of `g_R`, assembles their derivative matrices, and uses the
largest sampled operator norm times `lipschitz_factor=1.5`, floored at `1e-6`.
Finite probes and this margin cannot certify a global bound on general `G`.
This budget procedure is an implementation heuristic, not a condition or
parameter in the continuous mathematical objective.

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
rejected without evaluating the field there.

The adaptive estimate does not certify global accuracy. Taylor differentiation
can be expensive at high order and explicit time stepping can be problematic
for stiff fields. The exact local flow has identity and composition where the
solutions exist; `defect` measures only numerical composition error.

## D-006 — Training and persistence

Adam, uniform volume, fixed validation, exact spectral projection and cached
reference targets are fixed choices. Training exposes architecture widths,
manual or empirical Lipschitz budget, dataset size, minibatch size, epochs,
learning rate, seed, device and dtype. The best independent validation epoch
is restored. Metadata records `R`, `k`, normalization, training settings,
quadrature details, calibration method and exact NGF source revision.

Schema-5 checkpoints record the open-ball domain and `k` with model weights.
Loading checks reduced dimension, basis signature, radius, domain and Sobolev
order. The basis signature is a compatibility guard, not a content hash.
Older schemas 1–4 are read using their own field normalization, time scaling
and historical support rules, without changing their meaning. New training
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
spectral bound is reported separately. Empty radial bins have zero count and
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
last requested time. The time recorded by the solver is a numerical last
interior time, not the exact first exit. Optional time-step refinement is a
numerical indicator, not a certified error bound; a refined trajectory may
exit before its coarse counterpart.

## D-009 — Function-valued study and indexed derivative option

`Geometry`, `Space`, restrictions and weak-form operators exposed by GNS
refer to the exact NGF objects. `System(basis,weak,radius,sobolev_order,...)`
defines a study, `study.train(...)` produces a function-valued `Model`, and
`study.load(path)` reconstitutes a matching schema-5 model. The historical
`NeuralSemigroupProblem` and `NeuralSemigroup` remain available unchanged.
The new workflow requires an open-ball schema-5 model; it does not reinterpret
the historical time and taper semantics of schemas 1–4.

`model.state(u0)` projects a physical initial function, while
`model.from_coefficients(z)` explicitly accepts coordinates `[...,N]` strictly
inside the model's ball. These NGF `State`, `Function` and `Solution` types
perform the same spatial evaluation, componentwise integral and L² norm as
the numerical workflow. `model.evolve(initial,times,order,step|tolerance)`
uses the shared Taylor integrator and returns only completed requested times
if a numerical boundary exit occurs. `solution.exit_status()` then also
contains the last accepted interior state and time. `solution.at(t)` accepts
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
transferred without changing its values or sampling order. Exact spectral
projections of a training network are reused within a batch and its backward
pass; the associated graph is discarded before the next optimizer step.

`Model` initializes `ngfield.FunctionalFlow` with a compatible autonomous
field. Independent evaluation accepts strictly increasing or decreasing
time grids, including negative values, with the initial state at the first
requested time. Prefixes of paths that leave the ball come from NGF's
already accepted output states. Reports retain the requested order and their
normalized metrics; numerical exit times remain approximate.
