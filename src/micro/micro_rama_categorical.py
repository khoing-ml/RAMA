from __future__ import annotations

import torch
from torch import nn


class FeedForwardBlock(nn.Module):
    """Transformer feed-forward block with a residual connection."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention followed by a feed-forward block."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0, mlp_expansion: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ff = FeedForwardBlock(dim, expansion=mlp_expansion, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return self.ff(x + self.dropout(attn_out))


class CategoricalMicroRAMANet(nn.Module):
    """Predict categorical logits for quantized RAMA coordinates."""

    def __init__(
        self,
        context_dim: int = 256,
        patch_dim: int = 16,
        hidden_dim: int = 512,
        num_bins: int = 256,
        num_layers: int = 4,
        architecture: str = "mlp",
        dim_emb_dim: int = 64,
        transformer_dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if patch_dim <= 0:
            raise ValueError("patch_dim must be positive")
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than 1")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if architecture not in {"mlp", "transformer"}:
            raise ValueError(f"unsupported categorical micro architecture: {architecture}")

        self.context_dim = context_dim
        self.patch_dim = patch_dim
        self.num_bins = num_bins
        self.architecture = architecture

        if architecture == "transformer":
            model_dim = int(transformer_dim or hidden_dim)
            # Context vector projected to model_dim serves as the CLS token.
            self.context_proj = nn.Linear(context_dim, model_dim)
            # One learnable embedding per coord dim — analogous to ViT patch positional embeddings.
            self.coord_dim_embed = nn.Embedding(patch_dim, model_dim)
            # Stack of identical ViT encoder blocks (pre-norm self-attn + FFN).
            self.vit_blocks = nn.ModuleList(
                [SelfAttentionBlock(model_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
            )
            self.norm = nn.LayerNorm(model_dim)
            self.head = nn.Linear(model_dim, num_bins)
            return

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
        if self.architecture == "transformer":
            # CLS token: projected context, one per patch — (B*P, 1, model_dim)
            cls = self.context_proj(context).reshape(batch * num_patches, 1, -1)
            # Coord dim tokens: positional embeddings — (B*P, patch_dim, model_dim)
            coord_tokens = self.coord_dim_embed(dim_ids).unsqueeze(0).expand(batch * num_patches, -1, -1)
            # Sequence: [CLS, coord_0, ..., coord_{D-1}]
            tokens = torch.cat([cls, coord_tokens], dim=1)
            for block in self.vit_blocks:
                tokens = block(tokens)
            tokens = self.norm(tokens)
            # Drop CLS, predict from coord token outputs
            logits = self.head(tokens[:, 1:, :]).reshape(batch, num_patches, self.patch_dim, self.num_bins)
            if logits.shape != (batch, num_patches, self.patch_dim, self.num_bins):
                raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)}")
            return logits

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
        hidden_dim=int(config.get("hidden_dim", 512)),
        num_bins=int(num_bins or config.get("num_bins", 256)),
        num_layers=int(config.get("num_layers", 4)),
        architecture=str(config.get("architecture", "mlp")),
        dim_emb_dim=int(config.get("dim_emb_dim", 64)),
        transformer_dim=int(config.get("transformer_dim", config.get("hidden_dim", 512))),
        num_heads=int(config.get("num_heads", 4)),
        dropout=float(config.get("dropout", 0.0)),
    )
