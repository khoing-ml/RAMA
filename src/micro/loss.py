from __future__ import annotations

import torch
import torch.nn.functional as F


def categorical_micro_loss(logits: torch.Tensor, tokens: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Cross-entropy over quantized RAMA coordinate tokens."""
    if tokens.dtype != torch.long:
        raise TypeError(f"expected tokens dtype torch.long, got {tokens.dtype}")
    if logits.shape[:-1] != tokens.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and tokens {tuple(tokens.shape)} are incompatible")
    if logits.shape[-1] != num_bins:
        raise ValueError(f"expected logits last dim {num_bins}, got {logits.shape[-1]}")
    return F.cross_entropy(logits.reshape(-1, num_bins), tokens.reshape(-1))


def continuous_micro_nll_loss(eps: torch.Tensor, logabsdet: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood for the optional continuous spline-flow micro model."""
    log_base = -0.5 * (eps.square() + torch.log(torch.tensor(2.0 * torch.pi, device=eps.device, dtype=eps.dtype)))
    return -(log_base + logabsdet).mean()

