from __future__ import annotations

import torch

from src.modules.flow_matching import sample_heun


@torch.no_grad()
def sample_euler(
    model: torch.nn.Module,
    shape: tuple[int, int, int, int],
    num_steps: int = 50,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Sample macro latents with explicit Euler integration."""
    z = torch.randn(shape, device=device)
    batch_size = shape[0]
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((batch_size,), i / num_steps, device=device)
        z = z + dt * model(z, t)
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
    if method == "heun":
        return sample_heun(model, shape=shape, num_steps=num_steps, device=device)
    raise ValueError(f"unsupported sampler method: {method}")

