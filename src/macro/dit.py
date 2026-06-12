from __future__ import annotations

import torch
from torch import nn

from src.macro.unet import SinusoidalTimeEmbedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for 2D sin-cos embeddings, got {embed_dim}")
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    positions = positions.reshape(-1)
    out = torch.einsum("m,d->md", positions, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + gate_msa[:, None, :] * attn_out
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + gate_mlp[:, None, :] * mlp_out


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiTFlow(nn.Module):
    """DiT velocity model for low-frequency latent shortcut training."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        resolution: int = 16,
        patch_size: int = 2,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        time_embedding_dim: int | None = None,
        condition_on_dt: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if resolution % patch_size != 0:
            raise ValueError(f"resolution={resolution} must be divisible by patch_size={patch_size}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.condition_on_dt = condition_on_dt
        self.num_patches_side = resolution // patch_size
        self.num_patches = self.num_patches_side * self.num_patches_side

        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        cond_dim = int(time_embedding_dim or hidden_size)
        self.t_embedder = nn.Sequential(
            SinusoidalTimeEmbedding(cond_dim),
            nn.Linear(cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.dt_embedder = (
            nn.Sequential(
                SinusoidalTimeEmbedding(cond_dim),
                nn.Linear(cond_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            if condition_on_dt
            else None
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.num_patches_side)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(init_linear)
        nn.init.xavier_uniform_(self.x_embedder.weight.view(self.x_embedder.weight.shape[0], -1))
        if self.x_embedder.bias is not None:
            nn.init.constant_(self.x_embedder.bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        patch = self.patch_size
        channels = self.out_channels
        height = width = self.num_patches_side
        x = x.reshape(batch, height, width, patch, patch, channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(batch, channels, height * patch, width * patch)

    def forward(self, x: torch.Tensor, t: torch.Tensor, dt_base: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[-2:] != (self.resolution, self.resolution):
            raise ValueError(f"expected latent resolution {self.resolution}x{self.resolution}, got {tuple(x.shape[-2:])}")
        tokens = self.x_embedder(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype, device=tokens.device)
        cond = self.t_embedder(t)
        if self.dt_embedder is not None:
            if dt_base is None:
                dt_base = torch.zeros_like(t)
            cond = cond + self.dt_embedder(dt_base.to(device=t.device, dtype=t.dtype))
        for block in self.blocks:
            tokens = block(tokens, cond)
        return self.unpatchify(self.final_layer(tokens, cond))


def build_dit_flow(config: dict[str, object]) -> DiTFlow:
    input_shape = config.get("input_shape", [4, 16, 16])
    variant = str(config.get("variant", "")).lower()
    defaults = {
        "dit-s":   {"hidden_size": 384, "depth": 12, "num_heads": 6, "patch_size": 2},
        "dit-s/2": {"hidden_size": 384, "depth": 12, "num_heads": 6, "patch_size": 2},
        "dit-b":   {"hidden_size": 768, "depth": 12, "num_heads": 12, "patch_size": 2},
        "dit-b/2": {"hidden_size": 768, "depth": 12, "num_heads": 12, "patch_size": 2},
        "dit-b/4": {"hidden_size": 768, "depth": 12, "num_heads": 12, "patch_size": 4},
    }.get(variant, {})
    return DiTFlow(
        in_channels=int(config.get("in_channels", input_shape[0])),
        out_channels=int(config.get("out_channels", input_shape[0])),
        resolution=int(config.get("resolution", input_shape[-1])),
        patch_size=int(config.get("patch_size", defaults.get("patch_size", 2))),
        hidden_size=int(config.get("hidden_size", defaults.get("hidden_size", 768))),
        depth=int(config.get("depth", defaults.get("depth", 12))),
        num_heads=int(config.get("num_heads", defaults.get("num_heads", 12))),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        time_embedding_dim=int(config.get("time_embedding_dim", config.get("hidden_size", defaults.get("hidden_size", 768)))),
        condition_on_dt=bool(config.get("condition_on_dt", config.get("objective") == "shortcut")),
        dropout=float(config.get("dropout", 0.0)),
    )
