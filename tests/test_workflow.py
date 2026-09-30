"""The neural and numerical libraries expose the same function-valued path."""

import pytest
import torch
from ngfield import Function, Geometry, Space, State
from ngfield import System as NumericalSystem

from galerkin_neural_semigroup import Model, System
from tests.test_evaluation import reaction_model


def test_model_evolves_physical_functions_and_reconstructs_indexed_derivatives():
    neural = reaction_model(radius=2.0)
    model = Model(neural)
    assert isinstance(model, NumericalSystem)
    initial = model.state(lambda x: 0.2 * torch.ones_like(x))
    times = torch.tensor([0.0, 0.1, 0.2], dtype=model.dtype)
    solution = model.evolve(initial, times, step=0.04, order=3)
    state = solution.at(times[-1])
    points = torch.tensor([[0.1], [0.7]], dtype=model.dtype)

    assert isinstance(state, State) and state.basis is neural.basis
    torch.testing.assert_close(
        state.coefficients(), torch.exp(-times[-1]) * initial.coefficients(), atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(state.velocity().coefficients(), -state.coefficients())
    assert state.values(points).shape == (2, 1)
    assert state.gradient(points).shape == (2, 1, 1)
    assert state.hessian(points).shape == (2, 1, 1, 1)
    assert state.integral().shape == (1,)
    assert not solution.exit_status()["exited"]
    torch.testing.assert_close(
        solution.values(points), torch.stack([solution.at(time).values(points) for time in times])
    )

    indexed = model.indexed_derivatives(state, 2)
    assert tuple(indexed) == ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    assert all(isinstance(function, Function) for function in indexed.values())
    torch.testing.assert_close(indexed[(0, 0)].coefficients(), -state.coefficients())
    torch.testing.assert_close(
        indexed[(1, 0)].coefficients(), state.coefficients().new_tensor([-1, 0])
    )
    for alpha in ((2, 0), (1, 1), (0, 2)):
        torch.testing.assert_close(indexed[alpha].coefficients(), torch.zeros(2, dtype=model.dtype))
    torch.testing.assert_close(model.metrics.radial_rate(state), -state.norm_L2().square())
    torch.testing.assert_close(model.metrics.integral_rate(state), -state.integral())
    torch.testing.assert_close(
        model.metrics.sampled_lipschitz([state]), state.norm_L2().new_tensor(1.0)
    )


def test_model_reports_reserved_field_and_trajectory_metrics_by_method():
    model = Model(reaction_model())
    states = model.from_coefficients(torch.tensor([[0.2, 0.1], [0.3, -0.2]], dtype=model.dtype))
    field = model.evaluate.field(states, batch_size=1)
    assert field.rmse() < 1e-12 and field.sobolev() < 1e-12
    assert field.derivatives().shape == (1,)
    assert field.derivatives(by_order=False) < 1e-12
    assert int(field.radial()["counts"].sum()) == 2
    assert field.lipschitz()["reference_sample"] > 0

    times = torch.tensor([0.0, 0.1], dtype=model.dtype)
    report = model.evaluate.trajectories(states, times, step=0.02, refine=True)
    assert report.trajectories()["learned_states"].shape == (2, 2, 2)
    assert report.error()["mean"].max() < 1e-11
    assert report.field_on_paths().max() < 1e-11
    assert report.integrals()["gap_mean"].shape == (2, 1)
    assert report.norms()["radial_rate_gap_mean"].shape == (2,)
    assert report.exits()["common_count"].tolist() == [2, 2]
    assert torch.isfinite(report.time_refinement()["reference_refinement"]).all()
    with pytest.raises(ValueError, match="refine=True"):
        model.evaluate.trajectories(states, times).time_refinement()


def test_open_ball_exit_returns_only_completed_function_states():
    model = Model(reaction_model(radius=1.0, constant=True))
    initial = model.from_coefficients(torch.tensor([0.9, 0.0], dtype=model.dtype))
    times = torch.tensor([0.0, 0.05, 0.15], dtype=model.dtype)
    solution = model.evolve(initial, times, step=0.05)
    assert solution.exit_status()["exited"]
    assert solution.coefficients().shape == (2, 2)
    assert solution.exit_status()["last_accepted_state"].norm_L2() < 1
    with pytest.raises(ValueError, match="open ball"):
        model.from_coefficients(torch.tensor([1.0, 0.0], dtype=model.dtype))


def test_study_exports_same_space_vocabulary_and_loads_trained_model(tmp_path):
    geometry = Geometry(vertices=[[0.0], [1.0]], simplices=[[0, 1]])
    basis = Space(geometry=geometry, components=1).basis("polynomial", degree=0)
    study = System(
        basis=basis,
        weak=lambda u, v, dx, ds: -u[0] * v[0] * dx,
        radius=1.0,
        sobolev_order=0,
    )
    assert study.basis is basis and study.space is basis.space and study.geometry is geometry
    model = study.train(hidden=(2,), lipschitz=2.0, samples=8, batch_size=4, epochs=1, seed=17)
    assert model.training_history["training_loss"]
    path = tmp_path / "flow.gns"
    model.save(path)
    restored = study.load(path)
    state = restored.from_coefficients(torch.tensor([0.2], dtype=restored.dtype))
    torch.testing.assert_close(
        state.velocity().coefficients(),
        model.from_coefficients(state.coefficients()).velocity().coefficients(),
    )
