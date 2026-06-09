from __future__ import annotations

import torch
from torch import nn

from src.modules.unet_flow import group_norm
from src.modules.rama import rama_inverse, unpatchify

try:
    from nflows.transforms.splines import unconstrained_rational_quadratic_spline
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing.
    unconstrained_rational_quadratic_spline = None


class ContextEncoder(nn.Module):
    """Encode macro latents into one context vector per residual patch."""

    def __init__(
        self,
        in_channels: int = 4,
        context_dim: int = 256,
        hidden_channels: int = 128,
        num_layers: int = 3,
        use_position_embedding: bool = True,
        grid_size: tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        layers: list[nn.Module] = []
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
        self.net = nn.Sequential(*layers)
        self.context_dim = context_dim
        self.grid_size = grid_size
        self.position_embedding = (
            nn.Parameter(torch.zeros(1, grid_size[0] * grid_size[1], context_dim)) if use_position_embedding else None
        )

    def forward(self, z_l: torch.Tensor) -> torch.Tensor:
        if z_l.ndim != 4:
            raise ValueError(f"expected z_L shape [B, C, H, W], got {tuple(z_l.shape)}")
        context = self.net(z_l).flatten(2).transpose(1, 2)
        if self.position_embedding is not None:
            if context.shape[1] != self.position_embedding.shape[1]:
                raise ValueError(
                    f"expected {self.position_embedding.shape[1]} context positions, got {context.shape[1]}"
                )
            context = context + self.position_embedding
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
    return ContextEncoder(
        in_channels=int(config.get("in_channels", 4)),
        context_dim=int(config.get("context_dim", 256)),
        hidden_channels=int(config.get("hidden_channels", 128)),
        num_layers=int(config.get("num_layers", 3)),
        use_position_embedding=bool(config.get("positional_embedding", True)),
        grid_size=(int(grid[0]), int(grid[1])),
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
