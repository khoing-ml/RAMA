from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = group_norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.transpose(1, 2).reshape(batch, channels, height, width)
        return x + h


class UNetFlow(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        resolution: int = 16,
        base_channels: int = 128,
        channel_mult: list[int] | tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: list[int] | tuple[int, ...] = (8,),
        time_embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        channels = base_channels
        current_resolution = resolution
        self.down_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = []

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(channels, out_ch, time_embedding_dim))
                channels = out_ch
            attention = AttentionBlock(channels) if current_resolution in attention_resolutions else nn.Identity()
            self.down_levels.append(nn.ModuleDict({"blocks": blocks, "attention": attention}))
            skip_channels.append(channels)
            if level != len(channel_mult) - 1:
                self.downsamples.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
                current_resolution //= 2

        self.mid1 = ResBlock(channels, channels, time_embedding_dim)
        self.mid_attn = AttentionBlock(channels)
        self.mid2 = ResBlock(channels, channels, time_embedding_dim)

        self.up_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            blocks.append(ResBlock(channels + skip_channels[level], out_ch, time_embedding_dim))
            channels = out_ch
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(channels, out_ch, time_embedding_dim))
            attention = AttentionBlock(channels) if current_resolution in attention_resolutions else nn.Identity()
            self.up_levels.append(nn.ModuleDict({"blocks": blocks, "attention": attention}))
            if level != 0:
                self.upsamples.append(nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1))
                current_resolution *= 2

        self.output = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t)
        h = self.input(x)
        skips = []

        for level, down_level in enumerate(self.down_levels):
            for block in down_level["blocks"]:
                h = block(h, time_emb)
            h = down_level["attention"](h)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        h = self.mid1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, time_emb)

        for level, up_level in enumerate(self.up_levels):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in up_level["blocks"]:
                h = block(h, time_emb)
            h = up_level["attention"](h)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)

        return self.output(h)


def build_unet_flow(config: dict[str, object]) -> UNetFlow:
    input_shape = config.get("input_shape", [4, 16, 16])
    return UNetFlow(
        in_channels=int(config.get("in_channels", input_shape[0])),
        out_channels=int(config.get("out_channels", input_shape[0])),
        resolution=int(config.get("resolution", input_shape[-1])),
        base_channels=int(config.get("base_channels", 128)),
        channel_mult=list(config.get("channel_mult", [1, 2, 2])),
        num_res_blocks=int(config.get("num_res_blocks", 2)),
        attention_resolutions=list(config.get("attention_resolutions", [8])),
        time_embedding_dim=int(config.get("time_embedding_dim", 256)),
    )
