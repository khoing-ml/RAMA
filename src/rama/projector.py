from __future__ import annotations

import torch
from torch import nn


class RAMAProjector(nn.Module):
    """Module wrapper for frozen per-patch RAMA projection bases."""

    def __init__(self, bases: torch.Tensor) -> None:
        super().__init__()
        if bases.ndim != 3:
            raise ValueError(f"expected bases shape [P, d, d], got {tuple(bases.shape)}")
        if bases.shape[1] != bases.shape[2]:
            raise ValueError(f"expected square bases, got {tuple(bases.shape)}")
        self.register_buffer("bases", bases.float())

    @property
    def num_patches(self) -> int:
        return int(self.bases.shape[0])

    @property
    def patch_dim(self) -> int:
        return int(self.bases.shape[1])

    def project(self, patches: torch.Tensor) -> torch.Tensor:
        """Project flattened residual patches from [B, P, d] to RAMA coordinates."""
        if patches.ndim != 3:
            raise ValueError(f"expected patches shape [B, P, d], got {tuple(patches.shape)}")
        if patches.shape[1:] != self.bases.shape[:2]:
            raise ValueError(f"patches {tuple(patches.shape)} and bases {tuple(self.bases.shape)} are incompatible")
        return torch.einsum("bpd,pde->bpe", patches, self.bases)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Invert projected RAMA coordinates from [B, P, d] back to patches."""
        if y.ndim != 3:
            raise ValueError(f"expected y shape [B, P, d], got {tuple(y.shape)}")
        if y.shape[1:] != self.bases.shape[:2]:
            raise ValueError(f"projected values {tuple(y.shape)} and bases {tuple(self.bases.shape)} are incompatible")
        return torch.einsum("bpd,ped->bpe", y, self.bases)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.project(patches)

