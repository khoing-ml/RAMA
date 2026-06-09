from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_matching_loss(model: torch.nn.Module, z_l: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rectified-flow loss on macro latents."""
    batch_size = z_l.shape[0]
    z0 = torch.randn_like(z_l)
    z1 = z_l
    t = torch.rand(batch_size, device=z_l.device)
    t_view = t.view(batch_size, 1, 1, 1)

    z_t = (1.0 - t_view) * z0 + t_view * z1
    target_v = z1 - z0
    pred_v = model(z_t, t)
    loss = F.mse_loss(pred_v, target_v)
    metrics = {
        "loss": loss.detach(),
        "target_v_norm": target_v.detach().float().pow(2).mean().sqrt(),
        "pred_v_norm": pred_v.detach().float().pow(2).mean().sqrt(),
    }
    return loss, metrics


@torch.no_grad()
def sample_euler(
    model: torch.nn.Module,
    shape: tuple[int, int, int, int],
    num_steps: int = 50,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Sample macro latents by explicit Euler integration."""
    z = torch.randn(shape, device=device)
    batch_size = shape[0]
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((batch_size,), i / num_steps, device=device)
        z = z + dt * model(z, t)

    return z


@torch.no_grad()
def sample_heun(
    model: torch.nn.Module,
    shape: tuple[int, int, int, int],
    num_steps: int = 50,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Sample macro latents by integrating the learned vector field."""
    z = torch.randn(shape, device=device)
    batch_size = shape[0]
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((batch_size,), i / num_steps, device=device)
        t_next = torch.full((batch_size,), (i + 1) / num_steps, device=device)
        v = model(z, t)
        z_pred = z + dt * v
        v_next = model(z_pred, t_next)
        z = z + 0.5 * dt * (v + v_next)

    return z
