# Galerkin Neural Semigroup

Galerkin Neural Semigroup learns an autonomous reduced vector field from a complete
weak formulation and obtains time evolution by integrating that field. The public
workflow never exposes the numerical Galerkin field used as supervision:

```text
weak problem + fixed basis -> private reference evaluations -> neural field -> flow
```

The network is a `tanh` multilayer perceptron with exact spectral projection. If its
global Lipschitz budget is `L`, the autonomous ODE is globally well posed and its
continuous flow satisfies identity and composition by construction. Training compares
component-balanced field values and complete Jacobians on concentric radial layers; it
does not generate reference trajectories. Optional adaptive refinement inserts a new
spherical layer inside the adjacent radial interval with the largest measured error.

This repository is early research software. The mathematical and numerical contracts
are explicit, but empirical claims will be added only after dedicated experiments.

## Source installation

Python 3.11 or newer is required. Until the post-D-013 version of Numerical Galerkin
Field is released, the dependency is pinned to the exact source revision used here:

```bash
python -m pip install -e ".[dev]"
```

The distribution is named `galerkin-neural-semigroup`; its Python import is
`galerkin_neural_semigroup`.

## Compact API

The user prepares the geometry, admissible space and fixed operational basis with
[`ngfield`](https://github.com/cmurillos/Numerical-Galerkin-Field). The basis already
carries geometry, physical components and homogeneous restrictions, so these data are
not repeated in the learning problem.

```python
import numpy as np
import torch

from galerkin_neural_semigroup import NeuralSemigroupProblem
from ngfield import SimplicialDomain, Space, ZeroTrace, grad, inner


vertices = np.linspace(0, 1, 17)[:, None]
simplices = np.column_stack((np.arange(16), np.arange(1, 17)))
geometry = SimplicialDomain(vertices=vertices, simplices=simplices)
V = Space(
    geometry=geometry,
    components=1,
    restrictions=[ZeroTrace(component=0, boundary="all")],
)
basis = V.basis("laplacian", size=4, degree=1)


def weak(u, v, dx, ds):
    return -0.05 * inner(grad(u[0]), grad(v[0])) * dx


problem = NeuralSemigroupProblem(
    basis=basis,
    weak=weak,
    radius=1.5,
)

semigroup = problem.train(
    hidden=(64, 64),
    lipschitz=10.0,
    samples=10_000,
    batch_size=256,
    epochs=1_000,
    max_epochs=5_000,
    tolerance=1e-3,
    max_time=900,
    refine_every=200,
    refine_samples=512,
    candidate_samples=32_768,
    patience=100,
    lr=1e-3,
    jacobian_weight=0.1,
    balance_epsilon=1e-6,
    seed=0,
)
```

`basis.dimension` determines both the input and output dimensions. The initial design
uses equally spaced radii from `0` to `radius`; Gaussian directions are normalized to
the unit sphere before each positive radius is applied. The origin is retained as a
separate state. This concentric design, `tanh` activation, autonomous architecture,
spectral projection and Adam optimizer are method decisions rather than user-facing
objects. The loss balances every output component by its fixed inverse RMS scale on the
initial design and combines field values with the complete radius-scaled Jacobian in all
`N` canonical directions. `jacobian_weight` controls derivative supervision and
`balance_epsilon` places a relative floor under components with very small reference
energy. Independent value validation selects checkpoints and controls stopping; a small
periodic full-Jacobian audit is reported only as a diagnostic.

`epochs` is the minimum training budget and `max_epochs` is its hard epoch limit. If
`max_epochs` is omitted it equals `epochs`, preserving fixed-budget training. Training
can also stop when the 99th percentile of the independent probe score reaches
`tolerance`, when `max_time` seconds elapse, or when the fit remains stalled after the
minimum budget. The best checkpoint is retained and `stop_reason` records the decision.

Setting `refine_every` enables adaptive sampling. Once validation has failed to improve
for `patience` epochs, an independent probe measures the mean value error on a spherical
shell halfway between every pair of current adjacent layers. The midpoint shell with
the largest mean error defines the interval to refine. Refinement occurs only if the
probe's 99th-percentile value error exceeds the corresponding training error by at least
25%.
`refine_samples` new unit-sphere directions are then placed at the interval midpoint,
and later probes include that new layer. Field values and complete reference Jacobians
at training states remain private and cached. `candidate_checks` records the complete
radial error profile, while `refinements` records every inserted radius.

## Evolution and reconstruction

```python
z0 = semigroup.project(lambda x: torch.sin(torch.pi * x[:, :1]))
times = torch.linspace(0, 0.2, 21, dtype=semigroup.dtype, device=semigroup.device)

Z = semigroup.solve(z0, times)  # adaptive Dormand--Prince 5(4)
Z_rk4 = semigroup.solve(z0, times, step=1e-3)

points = torch.linspace(0, 1, 101, dtype=semigroup.dtype, device=semigroup.device).reshape(-1, 1)
U = semigroup.reconstruct(Z, points)
```

The learned scaled field is available as `semigroup.field(z)`. When `time_scale` is
not one, `semigroup.velocity(z)` returns the corresponding physical-time velocity.
The composition diagnostic

```python
defect = semigroup.defect(z0, 0.04, 0.07)
```

measures integration and roundoff error; the exact continuous flow itself satisfies
the composition law.

## Checkpoints

```python
semigroup.save("heat.gns")
restored = problem.load("heat.gns")
```

A checkpoint contains the neural parameters, architecture, training history and
reproducibility metadata. It deliberately does not serialize the private reference
evaluator. Loading therefore requires the same problem and operational basis.

## Advanced compatibility route

An explicit `ngfield.GalerkinProblem` remains usable for bases outside the modern
`Space` route:

```python
problem = NeuralSemigroupProblem.from_galerkin(
    galerkin_problem,
    basis=basis,
    radius=1.5,
)
```

This adapter still does not expose or ask the user to construct the reference field.

## Public contract

The stable surface is intentionally small:

| Object or operation | Meaning |
| --- | --- |
| `NeuralSemigroupProblem(...)` | Basis, weak form and reduced training domain. |
| `problem.train(...)` | Build private targets and fit the fixed neural architecture. |
| `NeuralSemigroup` | Trained field together with its continuous-flow interface. |
| `semigroup.field(z)` | Scaled learned autonomous field. |
| `semigroup.velocity(z)` | Learned velocity in physical time. |
| `semigroup.solve(z0, times)` | Reduced trajectory. |
| `semigroup.project` / `reconstruct` | Analysis and synthesis in the fixed basis. |
| `semigroup.save` / `problem.load` | Reproducible checkpoint round trip. |

Every leading state axis is a batch axis: the neural field maps `[...,N]` to
`[...,N]`.

See the [design contract](docs/design-contract.md) for fixed choices, invariants and
current limitations.

## Development

```bash
python -m pytest
ruff check .
ruff format --check .
python -m build
```

## Development provenance

The implementation was developed with substantial assistance from Astra, including
code generation and review. This assistance is declared explicitly; responsibility
for the method, validation and reported scientific results remains with the author.

## License

Copyright © 2026 Carlos Andrés Murillo. Distributed under the BSD 3-Clause License.
