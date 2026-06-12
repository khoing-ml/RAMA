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


@torch.no_grad()
def categorical_micro_metrics(logits: torch.Tensor, tokens: torch.Tensor, num_bins: int) -> dict[str, torch.Tensor]:
    """Metrics for categorical RAMA tokens, including near-miss accuracy."""
    if tokens.dtype != torch.long:
        raise TypeError(f"expected tokens dtype torch.long, got {tokens.dtype}")
    if logits.shape[:-1] != tokens.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and tokens {tuple(tokens.shape)} are incompatible")
    if logits.shape[-1] != num_bins:
        raise ValueError(f"expected logits last dim {num_bins}, got {logits.shape[-1]}")

    pred = logits.argmax(dim=-1)
    token_hist = torch.bincount(tokens.reshape(-1), minlength=num_bins).float()
    token_probs = token_hist / token_hist.sum().clamp_min(1.0)
    metrics = {
        "token_acc": (pred == tokens).float().mean(),
        "token_within_1": ((pred - tokens).abs() <= 1).float().mean(),
        "token_within_2": ((pred - tokens).abs() <= 2).float().mean(),
        "token_entropy": -(token_probs.clamp_min(1e-12) * token_probs.clamp_min(1e-12).log()).sum(),
        "token_clip_fraction": ((tokens == 0) | (tokens == num_bins - 1)).float().mean(),
    }
    for k in (5, 10):
        topk = min(k, num_bins)
        metrics[f"token_top{topk}_acc"] = (logits.topk(topk, dim=-1).indices == tokens.unsqueeze(-1)).any(dim=-1).float().mean()
    return metrics


def continuous_micro_nll_loss(eps: torch.Tensor, logabsdet: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood for the optional continuous spline-flow micro model."""
    log_base = -0.5 * (eps.square() + torch.log(torch.tensor(2.0 * torch.pi, device=eps.device, dtype=eps.dtype)))
    return -(log_base + logabsdet).mean()
