from __future__ import annotations

import torch
from torch import nn


class CategoricalMicroRAMANet(nn.Module):
    """Predict categorical logits for quantized RAMA coordinates."""

    def __init__(
        self,
        context_dim: int = 256,
        patch_dim: int = 16,
        dim_emb_dim: int = 64,
        hidden_dim: int = 512,
        num_bins: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if patch_dim <= 0:
            raise ValueError("patch_dim must be positive")
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than 1")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.context_dim = context_dim
        self.patch_dim = patch_dim
        self.num_bins = num_bins
        self.dim_embed = nn.Embedding(patch_dim, dim_emb_dim)

        layers: list[nn.Module] = []
        input_dim = context_dim + dim_emb_dim
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(hidden_dim, num_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [B, P, d, V]."""
        if context.ndim != 3:
            raise ValueError(f"expected context shape [B, P, D], got {tuple(context.shape)}")
        batch, num_patches, context_dim = context.shape
        if context_dim != self.context_dim:
            raise ValueError(f"expected context_dim={self.context_dim}, got {context_dim}")

        dim_ids = torch.arange(self.patch_dim, device=context.device)
        dim_emb = self.dim_embed(dim_ids)
        context_expanded = context[:, :, None, :].expand(batch, num_patches, self.patch_dim, context_dim)
        dim_emb = dim_emb[None, None, :, :].expand(batch, num_patches, self.patch_dim, -1)
        logits = self.net(torch.cat([context_expanded, dim_emb], dim=-1))
        if logits.shape != (batch, num_patches, self.patch_dim, self.num_bins):
            raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)}")
        return logits


def build_categorical_micro_rama_net(config: dict[str, object], num_bins: int | None = None) -> CategoricalMicroRAMANet:
    return CategoricalMicroRAMANet(
        context_dim=int(config.get("context_dim", 256)),
        patch_dim=int(config.get("patch_dim", 16)),
        dim_emb_dim=int(config.get("dim_emb_dim", 64)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        num_bins=int(num_bins or config.get("num_bins", 256)),
        num_layers=int(config.get("num_layers", 4)),
    )

