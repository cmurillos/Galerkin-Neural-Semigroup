"""Sampling and target preparation fixed by the learning functional."""

import torch


def uniform_ball(count, dimension, radius, *, generator, device, dtype):
    directions = torch.randn(
        count,
        dimension,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    while bool(torch.any(norms == 0).item()):
        mask = (norms == 0).squeeze(-1)
        directions[mask] = torch.randn(
            int(mask.sum().item()),
            dimension,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    radial = torch.rand(count, 1, generator=generator, device=device, dtype=dtype)
    radial = radius * radial.pow(1.0 / dimension)
    return radial * directions / norms


@torch.no_grad()
def reference_targets(reference, states, *, time_scale, batch_size):
    outputs = []
    for start in range(0, len(states), batch_size):
        values = reference(states[start : start + batch_size])
        if values.shape != states[start : start + batch_size].shape:
            raise ValueError("The internal reference evaluator changed the state shape.")
        if not torch.isfinite(values).all():
            raise FloatingPointError("The internal reference evaluator returned nonfinite values.")
        target = time_scale * values
        if not torch.isfinite(target).all():
            raise FloatingPointError("Time scaling produced nonfinite reference targets.")
        outputs.append(target.detach())
    return torch.cat(outputs, dim=0)
