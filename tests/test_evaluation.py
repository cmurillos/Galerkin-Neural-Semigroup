import numpy as np
import pytest
import torch
from ngfield import SimplicialDomain, Space

from galerkin_neural_semigroup import NeuralSemigroupProblem
from galerkin_neural_semigroup._network import _LocalBallField, _SpectralMLP
from galerkin_neural_semigroup.semigroup import NeuralSemigroup


def reaction_model(radius=2.0, *, constant=False, sobolev_order=1, components=1):
    geometry = SimplicialDomain(np.array([[0.0], [0.5], [1.0]]), np.array([[0, 1], [1, 2]]))
    basis = Space(geometry=geometry, components=components).basis(size=2)

    def weak(u, v, dx, ds):
        return -sum(u[i] * v[i] for i in range(components)) * dx

    problem = NeuralSemigroupProblem(
        basis=basis, weak=weak, radius=radius, sobolev_order=sobolev_order
    )
    reference = problem._build_coordinate_system(device=torch.device("cpu"), dtype=torch.float64)
    core = _SpectralMLP(2, (), 2.0, device=torch.device("cpu"), dtype=torch.float64)
    with torch.no_grad():
        core.weights[0].copy_(torch.zeros(2, 2) if constant else -torch.eye(2))
        core.biases[0].copy_(torch.tensor([1.0, 0.0]) if constant else torch.zeros(2))
    return NeuralSemigroup(
        field=_LocalBallField(core, radius, sobolev_order),
        coordinate_system=reference,
        radius=radius,
    )


def test_independent_field_error_reports_normalized_indexed_and_radial_metrics():
    model = reaction_model()
    x = torch.tensor([[0.1, 0.2], [0.9, 0.0], [-0.3, 0.1]], dtype=model.dtype)
    report = model.evaluate_field(model.radius * x, batch_size=2)
    assert report["field_rmse"] < 1e-12
    assert report["sobolev_rmse"] < 1e-12
    assert report["derivative_rmse_by_order"].shape == (2,)
    assert int(report["radial_counts"].sum()) == len(x)
    assert torch.isnan(report["radial_rmse"][report["radial_counts"] == 0]).all()
    torch.testing.assert_close(
        report["reference_lipschitz_sample"], torch.tensor(1.0, dtype=model.dtype)
    )
    torch.testing.assert_close(
        report["learned_lipschitz_sample"], torch.tensor(1.0, dtype=model.dtype)
    )
    assert report["learned_lipschitz_bound"] >= 1.0


def test_nonmatching_field_separates_value_derivative_and_sampled_sensitivity():
    model = reaction_model(constant=True, sobolev_order=0)
    x = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=model.dtype)
    report = model.evaluate_field(model.radius * x)
    torch.testing.assert_close(report["field_rmse"], x.new_tensor(1.625).sqrt())
    assert report["derivative_rmse_by_order"].shape == (1,)
    torch.testing.assert_close(report["reference_lipschitz_sample"], x.new_tensor(1.0))
    torch.testing.assert_close(report["learned_lipschitz_sample"], x.new_tensor(0.0))

    first_order = reaction_model(constant=True)
    report = first_order.evaluate_field(first_order.radius * x)
    torch.testing.assert_close(report["derivative_rmse_by_order"][1], x.new_tensor(2.0).sqrt())
    torch.testing.assert_close(report["sobolev_rmse"], x.new_tensor(3.625).sqrt())


def test_trajectory_report_matches_reaction_and_preserves_mass_change_semantics():
    model = reaction_model()
    initial = torch.tensor([[0.4, 0.2], [0.2, -0.3]], dtype=model.dtype)
    times = torch.tensor([0.0, 0.1, 0.2], dtype=model.dtype)
    report = model.evaluate_trajectories(initial, times, step=0.04, order=4, refine=True)
    expected = initial[None] / model.radius * torch.exp(-times[:, None, None])
    torch.testing.assert_close(report["reference_states"], expected, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(report["learned_states"], expected, atol=1e-8, rtol=1e-8)
    assert torch.max(report["trajectory_error_mean"]) < 1e-11
    assert torch.max(report["visited_field_rmse"]) < 1e-11
    assert torch.max(report["mass_gap_mean"]) < 1e-11
    assert torch.max(report["radial_rate_gap_mean"]) < 1e-11
    assert report["norm_change_reference"][-1] < 0
    assert torch.linalg.vector_norm(report["mass_change_reference"][-1]) > 0
    torch.testing.assert_close(report["reference_survival"], torch.ones_like(times))
    assert torch.isfinite(report["reference_refinement"]).all()
    assert torch.isfinite(report["learned_refinement"]).all()


def test_component_integrals_and_adaptive_refinement_keep_their_shapes():
    model = reaction_model(components=2)
    initial = torch.tensor([[0.2, 0.3], [0.1, -0.1]], dtype=model.dtype)
    times = torch.tensor([0.0, 0.01, 0.02], dtype=model.dtype)
    report = model.evaluate_trajectories(initial, times, tolerance=1e-8, refine=True)
    assert report["mass_gap_mean"].shape == (len(times), 2)
    assert report["mass_change_reference"].shape == (len(times), 2)
    assert report["mass_rate_gap_mean"].shape == (len(times), 2)
    assert torch.isfinite(report["learned_refinement"]).all()


def test_exit_statistics_keep_surviving_paths_and_never_evaluate_exterior():
    model = reaction_model(radius=1.0, constant=True)
    initial = torch.tensor([[0.9, 0.0], [0.1, 0.0]], dtype=model.dtype)
    times = torch.tensor([0.0, 0.05, 0.15, 0.3, 1.0], dtype=model.dtype)
    report = model.evaluate_trajectories(initial, times, step=0.05, refine=True)
    torch.testing.assert_close(
        report["learned_survival"], torch.tensor([1.0, 1.0, 0.5, 0.5, 0.0], dtype=model.dtype)
    )
    torch.testing.assert_close(report["reference_survival"], torch.ones_like(times))
    assert report["common_count"].tolist() == [2, 2, 1, 1, 0]
    assert torch.isnan(report["learned_states"][2:, 0]).all()
    assert 0.8 < report["learned_exit_time"][1] < 0.91
    assert 0.0 < report["learned_exit_time"][0] < 0.11
    assert torch.isfinite(report["trajectory_error_mean"][:-1]).all()
    assert torch.isnan(report["trajectory_error_mean"][-1])
    assert report["learned_refinement_count"][-1] == 0


def test_evaluation_rejects_boundary_states_and_ambiguous_times():
    model = reaction_model()
    with pytest.raises(ValueError, match="open ball"):
        model.evaluate_field(torch.tensor([[2.0, 0.0]], dtype=model.dtype))
    with pytest.raises(ValueError, match="times"):
        model.evaluate_trajectories(
            torch.tensor([[0.1, 0.0]], dtype=model.dtype),
            torch.tensor([0.1, 0.0, 0.2], dtype=model.dtype),
        )


@pytest.mark.parametrize("times", [[-0.2, -0.1, 0.0], [0.0, -0.1, -0.2]])
def test_evaluation_integrates_negative_times_in_either_direction(times):
    model = reaction_model()
    initial = torch.tensor([[0.2, 0.1], [-0.1, 0.3]], dtype=model.dtype)
    times = torch.tensor(times, dtype=model.dtype)
    report = model.evaluate_trajectories(initial, times, step=0.02, refine=True)
    expected = (initial / model.radius)[None] * torch.exp(-(times - times[0]))[:, None, None]
    torch.testing.assert_close(report["reference_states"], expected, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(report["learned_states"], expected, atol=1e-9, rtol=1e-9)
    assert torch.isfinite(report["reference_refinement"]).all()
    torch.testing.assert_close(
        model(initial[0], -0.2, step=0.02), initial[0] * torch.exp(times.new_tensor(0.2))
    )


def test_negative_time_exit_retains_prefix_and_survival():
    model = reaction_model(radius=1.0, constant=True)
    initial = torch.tensor([[-0.9, 0.0], [0.1, 0.0]], dtype=model.dtype)
    times = torch.tensor([0.0, -0.05, -0.15], dtype=model.dtype)
    report = model.evaluate_trajectories(initial, times, step=0.05, refine=True)
    torch.testing.assert_close(report["learned_survival"], times.new_tensor([1.0, 1.0, 0.5]))
    assert 0.0 > report["learned_exit_time"][0] > -0.11
    assert torch.isnan(report["learned_states"][2, 0]).all()
    assert torch.isfinite(report["learned_refinement"][:2, 0]).all()


def test_comparison_ends_at_reference_first_exit_and_backward_time_uses_maximum():
    model = reaction_model(radius=1.0, constant=True)
    initial = torch.tensor([[0.9, 0.0]], dtype=model.dtype)
    times = torch.tensor([0.0, -0.05, -0.15, -0.3], dtype=model.dtype)
    report = model.evaluate_trajectories(initial, times, step=0.02)
    assert report["common_count"].tolist() == [1, 1, 0, 0]
    torch.testing.assert_close(report["comparison_exit_time"], report["reference_exit_time"])
    assert torch.isnan(report["learned_exit_time"]).all()
    assert torch.isnan(report["trajectory_error_mean"][2:]).all()
    assert torch.isnan(report["visited_field_rmse"][2:]).all()
    assert torch.isnan(report["mass_rate_gap_mean"][2:]).all()
    assert torch.isnan(report["radial_rate_gap_mean"][2:]).all()
    assert report["reference_exit_state"].shape == initial.shape
    assert torch.linalg.vector_norm(report["reference_exit_state"][0]) == pytest.approx(1.0)
    assert report["reference_last_accepted_time"][0] > report["reference_exit_time"][0]
