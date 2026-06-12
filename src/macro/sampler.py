from __future__ import annotations

import math

import torch

from src.macro.losses import sample_heun


def _shortcut_dt_base(num_steps: int) -> float:
    if num_steps < 1 or num_steps & (num_steps - 1):
        raise ValueError(f"shortcut sampling requires a power-of-two --steps value, got {num_steps}")
    return float(int(math.log2(num_steps)))


def _model_velocity(
    model: torch.nn.Module,
    z: torch.Tensor,
    t: torch.Tensor,
    dt_base: torch.Tensor | None,
) -> torch.Tensor:
    if dt_base is not None or bool(getattr(model, "condition_on_dt", False)):
        return model(z, t, dt_base)
    return model(z, t)


@torch.no_grad()
def sample_euler(
    model: torch.nn.Module,
    shape: tuple[int, int, int, int],
    num_steps: int = 50,
    device: str | torch.device = "cuda",
    condition_on_dt: bool | None = None,
) -> torch.Tensor:
    """Sample macro latents with explicit Euler integration."""
    z = torch.randn(shape, device=device)
    batch_size = shape[0]
    dt = 1.0 / num_steps
    use_dt = bool(getattr(model, "condition_on_dt", False)) if condition_on_dt is None else condition_on_dt
    dt_base = None
    if use_dt:
        dt_base = torch.full((batch_size,), _shortcut_dt_base(num_steps), device=device)
    for i in range(num_steps):
        t = torch.full((batch_size,), i / num_steps, device=device)
        z = z + dt * _model_velocity(model, z, t, dt_base)
    return z


@torch.no_grad()
def sample_macro_latents(
    model: torch.nn.Module,
    shape: tuple[int, int, int, int],
    method: str = "heun",
    num_steps: int = 50,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Dispatch macro latent sampling to Euler or Heun."""
    if method == "euler":
        return sample_euler(model, shape=shape, num_steps=num_steps, device=device)
    if method == "shortcut":
        return sample_euler(model, shape=shape, num_steps=num_steps, device=device, condition_on_dt=True)
    if method == "heun":
        if bool(getattr(model, "condition_on_dt", False)):
            raise ValueError("shortcut-conditioned models should use method='shortcut' or method='euler'")
        return sample_heun(model, shape=shape, num_steps=num_steps, device=device)
    raise ValueError(f"unsupported sampler method: {method}")
