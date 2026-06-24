from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.macro.unet import group_norm
from src.modules.rama import rama_inverse, unpatchify

try:
    from nflows.transforms.splines import unconstrained_rational_quadratic_spline
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing.
    unconstrained_rational_quadratic_spline = None


# ---------------------------------------------------------------------------
# DDT-style DiT context encoder components (RMSNorm, 2D RoPE, gated FFN)
# ---------------------------------------------------------------------------

def _precompute_freqs_cis_2d(head_dim: int, height: int, width: int, theta: float = 10000.0, scale: float = 16.0) -> torch.Tensor:
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
    return freqs_cis.reshape(height * width, -1)  # [N, head_dim//2] complex


def _apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]  # [1, N, 1, head_dim//2] complex
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    q_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
    k_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
    return q_out.type_as(q), k_out.type_as(k)


class DiTContextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class DiTContextAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = DiTContextRMSNorm(self.head_dim)
        self.k_norm = DiTContextRMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, H, Hc]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = _apply_rotary_emb(q, k, freqs_cis)
        q = q.transpose(1, 2)   # [B, H, N, Hc]
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DiTContextFeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden_dim = int(2 * dim * mlp_ratio / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTContextBlock(nn.Module):
    """DDT-style pre-norm transformer block with RMSNorm, 2D RoPE, and SiLU-gated FFN."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = DiTContextRMSNorm(dim)
        self.attn = DiTContextAttention(dim, num_heads=num_heads)
        self.norm2 = DiTContextRMSNorm(dim)
        self.ff = DiTContextFeedForward(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------


class ContextFeedForwardBlock(nn.Module):
    """Feed-forward block used by the tiny ViT context encoder."""

    def __init__(self, dim: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = dim * mlp_ratio
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


class ContextTransformerBlock(nn.Module):
    """Pre-norm transformer block for context tokens."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ff = ContextFeedForwardBlock(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return self.ff(x + self.dropout(attn_out))


class ResConvBlock(nn.Module):
    """Pre-norm residual conv block: Norm -> SiLU -> Conv -> Norm -> SiLU -> Conv + skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ContextEncoder(nn.Module):
    """Encode macro latents into one context vector per residual patch."""

    def __init__(
        self,
        in_channels: int = 4,
        context_dim: int = 256,
        hidden_channels: int = 128,
        num_layers: int = 3,
        patch_size: int = 1,
        use_position_embedding: bool = True,
        grid_size: tuple[int, int] = (16, 16),
        architecture: str = "conv",
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if patch_size < 1:
            raise ValueError("patch_size must be at least 1")
        architecture = {
            "cnn": "conv", "tiny_vit": "vit", "transformer": "vit", "resnet": "resnet",
            "diffusion_transformer": "dit",
        }.get(architecture, architecture)
        if architecture not in {"conv", "vit", "resnet", "dit"}:
            raise ValueError(f"unsupported context encoder architecture: {architecture}")
        self.architecture = architecture
        self.context_dim = context_dim
        self.grid_size = grid_size
        # DiT uses 2D RoPE; other architectures use learned positional embeddings
        self.position_embedding = (
            nn.Parameter(torch.zeros(1, grid_size[0] * grid_size[1], context_dim))
            if use_position_embedding and architecture != "dit"
            else None
        )

        if architecture == "dit":
            if num_heads <= 0 or context_dim % num_heads != 0:
                raise ValueError(f"context_dim={context_dim} must be divisible by num_heads={num_heads}")
            self._patch_size = patch_size
            self.input_proj = nn.Linear(in_channels * patch_size * patch_size, context_dim)
            self.blocks = nn.ModuleList([
                DiTContextBlock(context_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(num_layers)
            ])
            self.output_norm = DiTContextRMSNorm(context_dim)
            head_dim = context_dim // num_heads
            freqs_cis = _precompute_freqs_cis_2d(head_dim, grid_size[0], grid_size[1])
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
            return

        if architecture == "vit":
            # patch_size > 1 merges z_L pixels into fewer, larger tokens
            self.input_proj = nn.Conv2d(in_channels, context_dim, kernel_size=patch_size, stride=patch_size)
            self.transformer_blocks = nn.Sequential(
                *[
                    ContextTransformerBlock(
                        context_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.output_norm = nn.LayerNorm(context_dim)
            return

        if architecture == "resnet":
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                *[ResConvBlock(hidden_channels) for _ in range(num_layers)],
                group_norm(hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, context_dim, kernel_size=1),
            ]
            if patch_size > 1:
                layers.append(nn.AvgPool2d(patch_size))
            self.net = nn.Sequential(*layers)
            return

        layers = []
        channels = in_channels
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
                    group_norm(hidden_channels),
                    nn.SiLU(),
                ]
            )
            channels = hidden_channels
        layers.append(nn.Conv2d(channels, context_dim, kernel_size=3, padding=1))
        if patch_size > 1:
            layers.append(nn.AvgPool2d(patch_size))
        self.net = nn.Sequential(*layers)

    def forward(self, z_l: torch.Tensor) -> torch.Tensor:
        if z_l.ndim != 4:
            raise ValueError(f"expected z_L shape [B, C, H, W], got {tuple(z_l.shape)}")
        if self.architecture == "dit":
            B, C, H, W = z_l.shape
            # patchify → embed → DiT blocks with 2D RoPE
            context = F.unfold(z_l, kernel_size=self._patch_size, stride=self._patch_size).transpose(1, 2)
            context = self.input_proj(context)
            for block in self.blocks:
                context = block(context, self.freqs_cis)
            return self.output_norm(context)
        if self.architecture == "vit":
            context = self.input_proj(z_l).flatten(2).transpose(1, 2)
        else:
            context = self.net(z_l).flatten(2).transpose(1, 2)
        if self.position_embedding is not None:
            if context.shape[1] != self.position_embedding.shape[1]:
                raise ValueError(
                    f"expected {self.position_embedding.shape[1]} context positions, got {context.shape[1]}"
                )
            context = context + self.position_embedding
        if self.architecture == "vit":
            context = self.output_norm(self.transformer_blocks(context))
        return context


class MicroRAMANet(nn.Module):
    """Conditional 1D rational-quadratic neural spline flow for RAMA coordinates."""

    def __init__(
        self,
        context_dim: int = 256,
        patch_dim: int = 16,
        dim_emb_dim: int = 64,
        hidden_dim: int = 512,
        spline_bins: int = 16,
        num_layers: int = 4,
        tail_bound: float = 3.0,
    ) -> None:
        super().__init__()
        if unconstrained_rational_quadratic_spline is None:
            raise ImportError("MicroRAMANet requires nflows. Install it with `pip install nflows`.")
        if spline_bins < 2:
            raise ValueError("spline_bins must be at least 2")
        if tail_bound <= 0:
            raise ValueError("tail_bound must be positive")
        self.patch_dim = patch_dim
        self.spline_bins = spline_bins
        self.tail_bound = tail_bound
        self.dim_embed = nn.Embedding(patch_dim, dim_emb_dim)

        layers: list[nn.Module] = []
        in_dim = 1 + context_dim + dim_emb_dim
        for layer_idx in range(num_layers):
            layer_in = in_dim if layer_idx == 0 else hidden_dim
            layers.append(ResidualMLPBlock(layer_in, hidden_dim) if layer_idx > 0 else nn.Sequential(
                nn.Linear(layer_in, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ))
        layers.append(nn.Linear(hidden_dim, 3 * spline_bins - 1))
        self.net = nn.Sequential(*layers)

    def _condition_inputs(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if y.ndim != 3:
            raise ValueError(f"expected y shape [B, P, d], got {tuple(y.shape)}")
        if context.ndim != 3:
            raise ValueError(f"expected context shape [B, P, D], got {tuple(context.shape)}")
        batch, num_patches, context_dim = context.shape
        if y.shape[:2] != (batch, num_patches):
            raise ValueError(f"y {tuple(y.shape)} and context {tuple(context.shape)} are incompatible")
        if y.shape[2] != self.patch_dim:
            raise ValueError(f"expected patch_dim={self.patch_dim}, got {y.shape[2]}")
        dim_ids = torch.arange(self.patch_dim, device=context.device)
        dim_emb = self.dim_embed(dim_ids)
        context = context[:, :, None, :].expand(batch, num_patches, self.patch_dim, context_dim)
        dim_emb = dim_emb[None, None, :, :].expand(batch, num_patches, self.patch_dim, -1)
        return torch.cat([y.unsqueeze(-1), context, dim_emb], dim=-1)

    def forward(self, y: torch.Tensor, context: torch.Tensor, inverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(self._condition_inputs(y, context))
        widths = params[..., : self.spline_bins]
        heights = params[..., self.spline_bins : 2 * self.spline_bins]
        derivatives = params[..., 2 * self.spline_bins :]
        outputs, logabsdet = unconstrained_rational_quadratic_spline(
            y,
            widths,
            heights,
            derivatives,
            inverse=inverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )
        return outputs, logabsdet


class ResidualMLPBlock(nn.Module):
    """Residual fully-connected block used by the conditional spline parameter net."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if in_dim != hidden_dim:
            raise ValueError("ResidualMLPBlock requires in_dim == hidden_dim")
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x + self.net(x))


def micro_nll_loss(eps: torch.Tensor, logabsdet: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood for the scalar flow target under a standard normal base."""
    log_base = -0.5 * (eps.square() + torch.log(torch.tensor(2.0 * torch.pi, device=eps.device, dtype=eps.dtype)))
    return -(log_base + logabsdet).mean()


def build_context_encoder(config: dict[str, object]) -> ContextEncoder:
    grid = tuple(config.get("grid_size", [16, 16]))
    # vit_layers/vit_heads/vit_mlp_ratio are old names kept for backwards compat
    num_layers = int(config.get("num_layers", config.get("vit_layers", 4)))
    num_heads = int(config.get("num_heads", config.get("vit_heads", 4)))
    mlp_ratio = int(config.get("mlp_ratio", config.get("vit_mlp_ratio", 4)))
    return ContextEncoder(
        in_channels=int(config.get("in_channels", 4)),
        context_dim=int(config.get("context_dim", 256)),
        hidden_channels=int(config.get("hidden_channels", 128)),
        num_layers=num_layers,
        patch_size=int(config.get("patch_size", 1)),
        use_position_embedding=bool(config.get("positional_embedding", True)),
        grid_size=(int(grid[0]), int(grid[1])),
        architecture=str(config.get("architecture", "conv")),
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=float(config.get("dropout", 0.0)),
    )


def build_micro_rama_net(config: dict[str, object]) -> MicroRAMANet:
    return MicroRAMANet(
        context_dim=int(config.get("context_dim", 256)),
        patch_dim=int(config.get("patch_dim", 16)),
        dim_emb_dim=int(config.get("dim_emb_dim", 64)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        spline_bins=int(config.get("spline_bins", config.get("num_bins", 16))),
        num_layers=int(config.get("num_layers", 4)),
        tail_bound=float(config.get("tail_bound", 3.0)),
    )


@torch.no_grad()
def sample_micro_latent(
    z_l: torch.Tensor,
    context_encoder: nn.Module,
    micro_model: nn.Module,
    bases: torch.Tensor,
    latent_channels: int = 4,
    latent_height: int = 32,
    latent_width: int = 32,
    patch_size: int = 2,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Sample a residual latent by inverting the conditional spline flow."""
    context_encoder.eval()
    micro_model.eval()

    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive")
    context = context_encoder(z_l)
    batch, num_patches, _ = context.shape
    patch_dim = bases.shape[-1]
    eps = noise_scale * torch.randn(batch, num_patches, patch_dim, device=z_l.device, dtype=z_l.dtype)
    y_hat, _ = micro_model(eps, context, inverse=True)
    patches_hat = rama_inverse(y_hat, bases)
    return unpatchify(
        patches_hat,
        channels=latent_channels,
        height=latent_height,
        width=latent_width,
        patch_size=patch_size,
    )
