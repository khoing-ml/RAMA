from __future__ import annotations

import torch


def patchify(z_h: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """Split residual latents into non-overlapping flattened patches."""
    if z_h.ndim != 4:
        raise ValueError(f"expected residual shape [B, C, H, W], got {tuple(z_h.shape)}")
    batch, channels, height, width = z_h.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"latent spatial size {height}x{width} must divide by patch_size={patch_size}")

    patches = z_h.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    return patches.reshape(batch, -1, channels * patch_size * patch_size)


def unpatchify(patches: torch.Tensor, channels: int, height: int, width: int, patch_size: int = 2) -> torch.Tensor:
    """Reassemble flattened patches into a residual latent map."""
    if patches.ndim != 3:
        raise ValueError(f"expected patches shape [B, P, d], got {tuple(patches.shape)}")
    batch, num_patches, patch_dim = patches.shape
    h_grid = height // patch_size
    w_grid = width // patch_size
    expected_dim = channels * patch_size * patch_size
    if num_patches != h_grid * w_grid:
        raise ValueError(f"expected {h_grid * w_grid} patches, got {num_patches}")
    if patch_dim != expected_dim:
        raise ValueError(f"expected patch_dim={expected_dim}, got {patch_dim}")

    x = patches.reshape(batch, h_grid, w_grid, channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(batch, channels, height, width)


def make_orthogonal_bases(
    num_patches: int,
    patch_dim: int,
    seed: int = 1234,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Create one deterministic orthogonal basis per patch position."""
    bases = []
    for patch_idx in range(num_patches):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + patch_idx)
        matrix = torch.randn(patch_dim, patch_dim, generator=generator, device=device)
        q, r = torch.linalg.qr(matrix)
        signs = torch.sign(torch.diag(r))
        signs[signs == 0] = 1
        bases.append(q * signs.unsqueeze(0))
    return torch.stack(bases, dim=0)


def rama_project(patches: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    """Project residual patches with per-position RAMA bases."""
    if patches.shape[1:] != bases.shape[:2]:
        raise ValueError(f"patches {tuple(patches.shape)} and bases {tuple(bases.shape)} are incompatible")
    return torch.einsum("bpd,pde->bpe", patches, bases)


def rama_inverse(y: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    """Invert per-position RAMA projections."""
    if y.shape[1:] != bases.shape[:2]:
        raise ValueError(f"projected values {tuple(y.shape)} and bases {tuple(bases.shape)} are incompatible")
    return torch.einsum("bpd,ped->bpe", y, bases)


def quantize(y: torch.Tensor, quant_bound: float, num_bins: int) -> torch.Tensor:
    """Convert projected scalar values into discrete bins."""
    if quant_bound <= 0:
        raise ValueError("quant_bound must be positive")
    y_norm = (y.clamp(-quant_bound, quant_bound) + quant_bound) / (2.0 * quant_bound)
    tokens = torch.floor(y_norm * num_bins).long()
    return tokens.clamp(0, num_bins - 1)


def dequantize(tokens: torch.Tensor, quant_bound: float, num_bins: int) -> torch.Tensor:
    """Convert discrete bins back to projected scalar bin centers."""
    if quant_bound <= 0:
        raise ValueError("quant_bound must be positive")
    return -quant_bound + (2.0 * quant_bound / num_bins) * (tokens.float() + 0.5)
