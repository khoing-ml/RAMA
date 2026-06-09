from __future__ import annotations

import torch
from torch import nn


class RAMATokenizer(nn.Module):
    """Uniform scalar tokenizer for RAMA-projected residual coordinates."""

    def __init__(self, num_bins: int = 256, bound: float = 3.0) -> None:
        super().__init__()
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than 1")
        if bound <= 0:
            raise ValueError("bound must be positive")
        self.num_bins = int(num_bins)
        self.bound = float(bound)

    def quantize(self, y: torch.Tensor) -> torch.Tensor:
        """Convert RAMA coordinates [B, P, d] into integer tokens."""
        y_norm = (y.clamp(-self.bound, self.bound) + self.bound) / (2.0 * self.bound)
        tokens = torch.floor(y_norm * self.num_bins).long()
        return tokens.clamp(0, self.num_bins - 1)

    def dequantize(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert integer tokens [B, P, d] back to bin-center RAMA coordinates."""
        if tokens.dtype != torch.long:
            raise TypeError(f"expected tokens dtype torch.long, got {tokens.dtype}")
        return -self.bound + (2.0 * self.bound / self.num_bins) * (tokens.float() + 0.5)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.quantize(y)

    def config_dict(self) -> dict[str, int | float]:
        return {"num_bins": self.num_bins, "bound": self.bound}


def load_tokenizer_config(path: str) -> dict[str, object]:
    config = torch.load(path, map_location="cpu")
    if not isinstance(config, dict):
        raise TypeError(f"tokenizer config at {path} must be a dict")
    if "num_bins" not in config or "bound" not in config:
        raise KeyError(f"tokenizer config at {path} must contain num_bins and bound")
    return config


def build_tokenizer_from_config(config: dict[str, object]) -> RAMATokenizer:
    return RAMATokenizer(num_bins=int(config["num_bins"]), bound=float(config["bound"]))

