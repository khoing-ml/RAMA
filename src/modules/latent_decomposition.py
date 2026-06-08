from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LatentDecomposition:
    z: torch.Tensor
    z_l: torch.Tensor
    z_l_up: torch.Tensor
    z_h: torch.Tensor


def decompose_latent(z: torch.Tensor) -> LatentDecomposition:
    """Split an SD-VAE latent into macro and residual components."""
    if z.ndim != 4:
        raise ValueError(f"expected latent shape [B, C, H, W], got {tuple(z.shape)}")
    if z.shape[-2] % 2 != 0 or z.shape[-1] % 2 != 0:
        raise ValueError(f"latent spatial size must be even, got {tuple(z.shape[-2:])}")

    ## this can be tested
    z_l = F.avg_pool2d(z, kernel_size=2, stride=2)
    z_l_up = F.interpolate(
        z_l,
        size=z.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    z_h = z - z_l_up
    return LatentDecomposition(z=z, z_l=z_l, z_l_up=z_l_up, z_h=z_h)


def reconstruct_from_decomposition(z_l: torch.Tensor, z_h: torch.Tensor) -> torch.Tensor:
    """Reconstruct the full latent from macro and residual components."""
    z_l_up = F.interpolate(
        z_l,
        size=z_h.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return z_l_up + z_h

