# Galerkin Neural Semigroup

Galerkin Neural Semigroup (GNS) trains an autonomous neural field against the
numerical Galerkin field supplied internally by
[Numerical Galerkin Field](https://github.com/cmurillos/Numerical-Galerkin-Field)
(NGF). Training uses values and **indexed state derivatives** of the fields on a
reduced coordinate ball. It does not use trajectories as targets or introduce
time as a network input.

This is early research software. Field errors on finite samples do not certify
trajectory accuracy or convergence to the original PDE.

## Install

Python 3.11 or newer:

```bash
python -m pip install -e ".[dev]"
```

The NGF dependency is pinned to an exact source commit until the new derivative
and integration APIs are released.

## Example

```python
import numpy as np
import torch
from ngfield import SimplicialDomain, Space, ZeroTrace, grad, inner
from galerkin_neural_semigroup import NeuralSemigroupProblem

vertices = np.linspace(0, 1, 17)[:, None]
simplices = np.column_stack((np.arange(16), np.arange(1, 17)))
geometry = SimplicialDomain(vertices, simplices)
space = Space(
    geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
)
basis = space.basis("laplacian", size=4, degree=1)


def weak(u, v, dx, ds):
    return -0.05 * inner(grad(u[0]), grad(v[0])) * dx


problem = NeuralSemigroupProblem(basis=basis, weak=weak, radius=1.5, sobolev_order=1)
model = problem.train(
    hidden=(64, 64), lipschitz="auto", samples=10_000, batch_size=256, epochs=1_000, lr=1e-3, seed=0
)
z0 = model.project(lambda x: torch.sin(torch.pi * x[:, :1]))
times = torch.linspace(0, 0.2, 21, dtype=model.dtype, device=model.device)
Z = model.solve(z0, times, order=4)
points = torch.linspace(0, 1, 101, dtype=model.dtype, device=model.device)[:, None]
U = model.reconstruct(Z, points)
```

The basis fixes the reduced dimension `N`, admissible physical space and component
shapes. GNS constructs the Galerkin reference privately. `radius=R` is the
**reduced-coordinate** radius; for an L²-orthonormal basis it also represents the
L² norm of the reconstructed reduced state. It is not a radius in physical space.
Changing `R` changes the target function in general, so it normally requires
training again.

## Loss and normalization

Write `z=Rx`, with `x` in the open unit ball `B_N(1)`. For the numerical Galerkin
field `G: ℝ^N → ℝ^N` and the neural field `fθ` on `B_N(1)`, define

```text
g_R(x) = G(Rx)/R,                         Fθ(z) = R fθ(z/R),
∂_x^α g_R(x) = R^(|α|-1) ∂_z^α G(Rx).
```

The objective approximates the usual integer Sobolev norm by a fixed sample:

```text
||fθ-g_R||²_H^k(B_N(1))
  = ∫_{B_N(1)} Σ_{|α|≤k} ||∂_x^α fθ(x) - ∂_x^α g_R(x)||²₂ dx,

L_B(θ) = (1/B) Σ_{i=1}^B Σ_{|α|≤k}
         ||∂_x^α fθ(x_i) - ∂_x^α g_R(x_i)||²₂.
```

Each multi-index occurs once; there are no factorial or ad hoc derivative weights.
The empirical loss estimates the integral divided by the volume of the ball.
NGF computes the reference derivatives and the indexed list. GNS caches detached
reference targets; its network derivatives remain differentiable with respect to
network parameters. The physical state space remains the L²-based Galerkin space:
`H^k` here concerns **derivatives with respect to reduced state coordinates**, not
spatial Sobolev regularity of the PDE solution.

Training and validation draw independent fixed samples from normalized volume on
the ball. If `ξ` is Gaussian and `s` uniform on `(0,1)`, then
`x=s^(1/N) ξ/||ξ||₂`. Adam trains the `tanh` MLP with exact spectral projection;
validation selects the best epoch. The `lipschitz="auto"` budget estimates
`||Dg_R||₂` at sampled points using indexed derivatives. Its optional margin is
heuristic, not a certified bound for `G`; a positive numeric budget may be supplied.

## Local flow and comparable integration

`model.field(z)` and `model.velocity(z)` both give the learned physical-time
field `Fθ(z)`. They are defined only when `||z||₂<R`. They raise at the boundary
or outside it. `solve` integrates locally and raises `ngfield.DomainExitError`
when a numerical step reaches the boundary. The error records the last accepted
interior state and time; it does not certify the exact exit instant. No value of
`Fθ` outside the ball is evaluated.

Both `G.solve` and `model.solve` call **the same NGF Taylor-jet integrator** in
physical time. Use the same `times`, `order`, `step` or `tolerance`, and optionally
`radius=R` on `G.solve`, for temporal comparisons:

```python
Z = model.solve(z0, times, order=4, step=1e-3)  # fixed maximum step
Z = model.solve(z0, times, order=4, tolerance=1e-8)  # adaptive step
```

The integrator uses `J₁[X]=X` and `J_{r+1}[X]=DJ_r[X]·X` to form the Taylor
polynomial of degree `p=order`; adaptive stepping estimates the first omitted
term. The time order `p` and Sobolev training order `k` are independent choices.
The adaptive estimate is a heuristic, not a rigorous global error bound. Explicit
Taylor integration may be inefficient or fail for stiff dynamics. The exact
local flow has identity and composition where defined; `model.defect(z,s,t)`
is a numerical integration diagnostic.

## Persistence and older models

```python
model.save("heat.gns")
restored = problem.load("heat.gns")
```

Schema 5 stores the open domain and Sobolev order along with network state,
training metadata and basis signature. Loading requires a compatible problem and
basis; the signature cannot prove arbitrary basis functions are identical. Schemas
1–4 remain loadable with their **historical** time scaling and support rules.
New models do not have a time scale or an exterior taper.

For an explicit older `ngfield.GalerkinProblem`, use
`NeuralSemigroupProblem.from_galerkin(problem, basis=basis, radius=R,
sobolev_order=k)`; the reference evaluator remains private.

## Validation and development

Compare the learned field with the Galerkin field on independent states, and
compare trajectories using matched integrator settings. Separately report
projection, Galerkin truncation, neural field, and time integration errors.

```bash
python -m pytest
ruff check .
ruff format --check .
python -m build
```

See [design contract](docs/design-contract.md), [agent guide](AGENTS.md),
[example](examples/heat.py) and [changelog](CHANGELOG.md).

## Provenance and license

Development involved substantial assistance from Astra, including code generation
and review. Responsibility for the method and scientific claims remains with the
author. Copyright © 2026 Carlos Andrés Murillo. BSD 3-Clause License.
