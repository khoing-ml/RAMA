from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _call_velocity_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    dt_base: torch.Tensor | None = None,
) -> torch.Tensor:
    if dt_base is not None or bool(getattr(model, "condition_on_dt", False)):
        return model(x, t, dt_base)
    return model(x, t)


def flow_matching_loss(model: torch.nn.Module, z_l: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rectified-flow loss on macro latents."""
    batch_size = z_l.shape[0]
    z0 = torch.randn_like(z_l)
    z1 = z_l
    t = torch.rand(batch_size, device=z_l.device)
    t_view = t.view(batch_size, 1, 1, 1)

    z_t = (1.0 - t_view) * z0 + t_view * z1
    target_v = z1 - z0
    pred_v = _call_velocity_model(model, z_t, t)
    loss = F.mse_loss(pred_v, target_v)
    metrics = {
        "loss": loss.detach(),
        "target_v_norm": target_v.detach().float().pow(2).mean().sqrt(),
        "pred_v_norm": pred_v.detach().float().pow(2).mean().sqrt(),
    }
    return loss, metrics


def _validate_denoise_timesteps(denoise_timesteps: int) -> int:
    if denoise_timesteps < 2 or denoise_timesteps & (denoise_timesteps - 1):
        raise ValueError(f"denoise_timesteps must be a power of two >= 2, got {denoise_timesteps}")
    return int(math.log2(denoise_timesteps))


def shortcut_matching_loss(
    model: torch.nn.Module,
    z_l: torch.Tensor,
    denoise_timesteps: int = 128,
    bootstrap_every: int = 8,
    bootstrap_dt_bias: float = 0.0,
    clip_intermediate: float = 4.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Shortcut-model loss on low-frequency latents.

    The flow subset learns the tiny-step vector field at dt=1/denoise_timesteps.
    The bootstrap subset learns larger shortcuts by composing two half-size
    predictions from the current model, matching the shortcut-model objective.
    """
    batch_size = z_l.shape[0]
    if batch_size < 2:
        raise ValueError("shortcut training needs batch size >= 2 so flow and bootstrap subsets are both non-empty")
    if bootstrap_every < 2:
        raise ValueError(f"bootstrap_every must be >= 2, got {bootstrap_every}")

    log2_sections = _validate_denoise_timesteps(denoise_timesteps)
    bootstrap_size = max(1, batch_size // bootstrap_every)
    bootstrap_size = min(bootstrap_size, batch_size - 1)
    flow_size = batch_size - bootstrap_size
    eps = 1.0e-5

    flow_z1 = z_l[bootstrap_size:]
    flow_z0 = torch.randn_like(flow_z1)
    flow_t = torch.randint(0, denoise_timesteps, (flow_size,), device=z_l.device, dtype=torch.int64).to(z_l.dtype)
    flow_t = flow_t / denoise_timesteps
    flow_t_view = flow_t.view(flow_size, 1, 1, 1)
    flow_xt = (1.0 - (1.0 - eps) * flow_t_view) * flow_z0 + flow_t_view * flow_z1
    flow_target = flow_z1 - (1.0 - eps) * flow_z0
    flow_dt_base = torch.full((flow_size,), float(log2_sections), device=z_l.device, dtype=z_l.dtype)

    bootstrap_z1 = z_l[:bootstrap_size]
    bootstrap_z0 = torch.randn_like(bootstrap_z1)
    if bootstrap_dt_bias > 0:
        high_count = bootstrap_size // 4
        mid_count = bootstrap_size // 4
        random_count = bootstrap_size - high_count - mid_count
        random_base = torch.randint(
            0,
            max(log2_sections - 1, 1),
            (random_count,),
            device=z_l.device,
            dtype=torch.int64,
        )
        bootstrap_dt_base_int = torch.cat(
            [
                random_base,
                torch.ones(mid_count, device=z_l.device, dtype=torch.int64),
                torch.zeros(high_count, device=z_l.device, dtype=torch.int64),
            ],
            dim=0,
        )
    else:
        bootstrap_dt_base_int = torch.randint(
            0,
            log2_sections,
            (bootstrap_size,),
            device=z_l.device,
            dtype=torch.int64,
        )

    dt_sections = 2**bootstrap_dt_base_int
    bootstrap_t_int = torch.floor(torch.rand(bootstrap_size, device=z_l.device) * dt_sections.float()).to(torch.int64)
    bootstrap_t = bootstrap_t_int.to(z_l.dtype) / dt_sections.to(z_l.dtype)
    bootstrap_t_view = bootstrap_t.view(bootstrap_size, 1, 1, 1)
    bootstrap_xt = (1.0 - (1.0 - eps) * bootstrap_t_view) * bootstrap_z0 + bootstrap_t_view * bootstrap_z1
    bootstrap_dt_base = bootstrap_dt_base_int.to(z_l.dtype)
    half_dt_base = bootstrap_dt_base + 1.0
    half_dt = (1.0 / (2.0 ** half_dt_base)).view(bootstrap_size, 1, 1, 1)

    with torch.no_grad():
        v_b1 = _call_velocity_model(model, bootstrap_xt, bootstrap_t, half_dt_base)
        bootstrap_t2 = bootstrap_t + half_dt.flatten()
        bootstrap_xt2 = bootstrap_xt + half_dt * v_b1
        if clip_intermediate > 0:
            bootstrap_xt2 = bootstrap_xt2.clamp(-clip_intermediate, clip_intermediate)
        v_b2 = _call_velocity_model(model, bootstrap_xt2, bootstrap_t2, half_dt_base)
        bootstrap_target = 0.5 * (v_b1 + v_b2)
        if clip_intermediate > 0:
            bootstrap_target = bootstrap_target.clamp(-clip_intermediate, clip_intermediate)

    train_xt = torch.cat([bootstrap_xt, flow_xt], dim=0)
    train_t = torch.cat([bootstrap_t, flow_t], dim=0)
    train_dt_base = torch.cat([bootstrap_dt_base, flow_dt_base], dim=0)
    target_v = torch.cat([bootstrap_target, flow_target], dim=0)
    pred_v = _call_velocity_model(model, train_xt, train_t, train_dt_base)

    per_item_mse = (pred_v - target_v).pow(2).mean(dim=(1, 2, 3))
    loss_bootstrap = per_item_mse[:bootstrap_size].mean()
    loss_flow = per_item_mse[bootstrap_size:].mean()
    loss = per_item_mse.mean()
    metrics = {
        "loss": loss.detach(),
        "loss_flow": loss_flow.detach(),
        "loss_bootstrap": loss_bootstrap.detach(),
        "bootstrap_ratio": torch.tensor(bootstrap_size / batch_size, device=z_l.device),
        "target_v_norm": target_v.detach().float().pow(2).mean().sqrt(),
        "pred_v_norm": pred_v.detach().float().pow(2).mean().sqrt(),
        "shortcut_dt_base_mean": train_dt_base.detach().float().mean(),
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
        z = z + dt * _call_velocity_model(model, z, t)

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
        v = _call_velocity_model(model, z, t)
        z_pred = z + dt * v
        v_next = _call_velocity_model(model, z_pred, t_next)
        z = z + 0.5 * dt * (v + v_next)

    return z
