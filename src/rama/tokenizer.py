from __future__ import annotations

import torch
from torch import nn


class RAMATokenizer(nn.Module):
    """Scalar tokenizer for RAMA-projected residual coordinates.

    Bins uniformly over [-bound, bound] by default. If `edges` and
    `bin_centers` are supplied (from quantile calibration), bins are
    non-uniform instead — narrower where the calibration data is dense,
    wider in the tails — which fits the heavy-tailed y distribution better
    than fixed-width bins at the same num_bins.
    """

    def __init__(
        self,
        num_bins: int = 256,
        bound: float = 3.0,
        edges: torch.Tensor | None = None,
        bin_centers: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than 1")
        if bound <= 0:
            raise ValueError("bound must be positive")
        self.num_bins = int(num_bins)
        self.bound = float(bound)

        if (edges is None) != (bin_centers is None):
            raise ValueError("edges and bin_centers must be provided together")
        if edges is not None:
            edges = edges.float().flatten()
            bin_centers = bin_centers.float().flatten()
            if edges.numel() != self.num_bins - 1:
                raise ValueError(f"expected {self.num_bins - 1} edges, got {edges.numel()}")
            if bin_centers.numel() != self.num_bins:
                raise ValueError(f"expected {self.num_bins} bin_centers, got {bin_centers.numel()}")
            self.register_buffer("edges", edges)
            self.register_buffer("bin_centers", bin_centers)
        else:
            self.edges = None
            self.bin_centers = None

    @property
    def quantile_based(self) -> bool:
        return self.edges is not None

    def quantize(self, y: torch.Tensor) -> torch.Tensor:
        """Convert RAMA coordinates [B, P, d] into integer tokens."""
        y = y.clamp(-self.bound, self.bound)
        if self.edges is not None:
            tokens = torch.bucketize(y.contiguous(), self.edges)
            return tokens.clamp(0, self.num_bins - 1)
        y_norm = (y + self.bound) / (2.0 * self.bound)
        tokens = torch.floor(y_norm * self.num_bins).long()
        return tokens.clamp(0, self.num_bins - 1)

    def dequantize(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert integer tokens [B, P, d] back to bin-center RAMA coordinates."""
        if tokens.dtype != torch.long:
            raise TypeError(f"expected tokens dtype torch.long, got {tokens.dtype}")
        if self.bin_centers is not None:
            return self.bin_centers[tokens]
        return -self.bound + (2.0 * self.bound / self.num_bins) * (tokens.float() + 0.5)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.quantize(y)

    def config_dict(self) -> dict[str, object]:
        config: dict[str, object] = {"num_bins": self.num_bins, "bound": self.bound}
        if self.edges is not None:
            config["edges"] = self.edges.cpu()
            config["bin_centers"] = self.bin_centers.cpu()
        return config


def load_tokenizer_config(path: str) -> dict[str, object]:
    config = torch.load(path, map_location="cpu")
    if not isinstance(config, dict):
        raise TypeError(f"tokenizer config at {path} must be a dict")
    if "num_bins" not in config or "bound" not in config:
        raise KeyError(f"tokenizer config at {path} must contain num_bins and bound")
    return config


def build_tokenizer_from_config(config: dict[str, object]) -> RAMATokenizer:
    return RAMATokenizer(
        num_bins=int(config["num_bins"]),
        bound=float(config["bound"]),
        edges=config.get("edges"),
        bin_centers=config.get("bin_centers"),
    )

