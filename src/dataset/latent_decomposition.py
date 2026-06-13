from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentDecomposition:
    z: torch.Tensor
    z_l: torch.Tensor
    z_l_up: torch.Tensor
    z_h: torch.Tensor


def decompose_latent(z: torch.Tensor) -> LatentDecomposition:
    """Split a latent into LL (macro) and {LH, HL, HH} (high-freq) subbands via Haar DWT.

    z_l  = LL subband  — identical to avg_pool2d(z, 2); shape [B, C, H/2, W/2]
    z_h  = cat([LH, HL, HH], dim=1)              ;  shape [B, C*3, H/2, W/2]
    z_l_up = IDWT(LL, 0, 0, 0) = nearest-neighbour upsample of z_l
    """
    if z.ndim != 4:
        raise ValueError(f"expected latent shape [B, C, H, W], got {tuple(z.shape)}")
    if z.shape[-2] % 2 != 0 or z.shape[-1] % 2 != 0:
        raise ValueError(f"latent spatial size must be even, got {tuple(z.shape[-2:])}")

    lo_rows = (z[:, :, 0::2, :] + z[:, :, 1::2, :]) * 0.5
    hi_rows = (z[:, :, 0::2, :] - z[:, :, 1::2, :]) * 0.5

    z_l = (lo_rows[:, :, :, 0::2] + lo_rows[:, :, :, 1::2]) * 0.5  # LL
    lh  = (lo_rows[:, :, :, 0::2] - lo_rows[:, :, :, 1::2]) * 0.5  # LH
    hl  = (hi_rows[:, :, :, 0::2] + hi_rows[:, :, :, 1::2]) * 0.5  # HL
    hh  = (hi_rows[:, :, :, 0::2] - hi_rows[:, :, :, 1::2]) * 0.5  # HH

    z_h = torch.cat([lh, hl, hh], dim=1)
    z_l_up = z_l.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    return LatentDecomposition(z=z, z_l=z_l, z_l_up=z_l_up, z_h=z_h)


def reconstruct_from_decomposition(z_l: torch.Tensor, z_h: torch.Tensor) -> torch.Tensor:
    """Reconstruct the full latent from macro (z_l) and high-freq (z_h) via Haar IDWT."""
    C = z_l.shape[1]
    lh, hl, hh = z_h[:, :C], z_h[:, C : 2 * C], z_h[:, 2 * C :]

    B, _, H, W = z_l.shape
    z = torch.empty(B, C, H * 2, W * 2, device=z_l.device, dtype=z_l.dtype)
    z[:, :, 0::2, 0::2] = z_l + lh + hl + hh
    z[:, :, 0::2, 1::2] = z_l - lh + hl - hh
    z[:, :, 1::2, 0::2] = z_l + lh - hl - hh
    z[:, :, 1::2, 1::2] = z_l - lh - hl + hh
    return z


def reconstruct_low_freq(z_l: torch.Tensor) -> torch.Tensor:
    """Return a full-resolution latent using only the low-frequency component (zero HF subbands).

    Equivalent to Haar IDWT with LH=HL=HH=0, which gives nearest-neighbour upsampling.
    Useful for visualising the macro-only reconstruction via a VAE decoder.
    """
    return z_l.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
