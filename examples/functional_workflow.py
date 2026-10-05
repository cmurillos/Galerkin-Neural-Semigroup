"""Train a small function-valued learned flow and evaluate reserved states."""

import torch

import galerkin_neural_semigroup as gns


def main():
    geometry = gns.Geometry(vertices=[[0.0], [1.0]], simplices=[[0, 1]])
    V = gns.Space(geometry=geometry, components=1)
    basis = V.basis("polynomial", degree=1)

    def weak(u, v, dx, ds):
        return -u[0] * v[0] * dx

    study = gns.System(basis=basis, weak=weak, radius=1.0, sobolev_order=1)
    model = study.train(hidden=(4,), samples=32, batch_size=8, epochs=3, seed=7)
    initial = model.state(lambda x: 0.2 * torch.ones_like(x))
    times = torch.tensor([0.0, 0.05, 0.1], dtype=model.dtype, device=model.device)
    path = model.evolve(initial, times, order=2, step=0.01)
    u_t = path.at(path.times[-1])
    print("local-flow status:", path.exit_status())
    points = torch.tensor([[0.2], [0.8]], dtype=model.dtype, device=model.device)
    print("learned u(t) at points:", u_t.values(points))
    print("indexed state derivatives:", tuple(u_t.indexed_derivatives(1)))

    print(
        "exterior-zero convention:",
        model.velocity_zero_extended(
            torch.tensor([model.radius, 0.0], dtype=model.dtype, device=model.device)
        ),
    )

    reserved = model.from_coefficients(
        torch.tensor([[0.1, 0.2], [-0.2, 0.1]], dtype=model.dtype, device=model.device)
    )
    print("held-out field RMS:", model.evaluate.field(reserved).rmse())
    print("held-out flow RMS:", model.evaluate.trajectories(reserved, times).error()["mean"])


if __name__ == "__main__":
    main()
