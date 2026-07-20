from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.latent_dataset import CachedMicroLatentDataset
from src.modules.rama import make_orthogonal_bases, patchify
from src.rama.projector import RAMAProjector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate RAMA tokenizer clipping bound from cached latents.")
    parser.add_argument("--latent-cache", default="data/latents")
    parser.add_argument("--bases", default="cache/rama_bases_p256_d16.pt")
    parser.add_argument("--output", default="cache/rama_tokenizer_config.pt")
    parser.add_argument("--num-bins", type=int, default=256)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument(
        "--mode",
        choices=("quantile", "uniform"),
        default="quantile",
        help="quantile: non-uniform bin edges fit to the calibration data's distribution "
        "(better for heavy-tailed y). uniform: fixed-width bins over [-bound, bound].",
    )
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--num-patches", type=int, default=256)
    parser.add_argument("--patch-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_or_make_bases(path: Path, num_patches: int, patch_dim: int, seed: int) -> torch.Tensor:
    if path.exists():
        return torch.load(path, map_location="cpu").float()
    bases = make_orthogonal_bases(num_patches=num_patches, patch_dim=patch_dim, seed=seed, device="cpu")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bases, path)
    return bases.float()


@torch.no_grad()
def collect_projected_values(
    dataloader: DataLoader,
    projector: RAMAProjector,
    patch_size: int = 2,
    max_batches: int = 200,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for step, batch in enumerate(dataloader):
        if step >= max_batches:
            break
        z_h = batch["z_H"].to(device)
        patches = patchify(z_h, patch_size=patch_size)
        y = projector.project(patches)
        values.append(y.flatten().cpu())
    if not values:
        raise ValueError("no values collected; check latent cache and max_batches")
    return torch.cat(values, dim=0)


def estimate_bound(all_values: torch.Tensor, percentile: float) -> float:
    return float(torch.quantile(all_values.abs(), percentile / 100.0).item())


def _centroids_for_edges(
    clipped: torch.Tensor,
    edges: torch.Tensor,
    bound: float,
    num_bins: int,
) -> torch.Tensor:
    tokens = torch.bucketize(clipped, edges).clamp(0, num_bins - 1)
    counts = torch.bincount(tokens, minlength=num_bins).float()
    sums = torch.zeros(num_bins, dtype=torch.float32).scatter_add_(0, tokens, clipped)
    centers = torch.zeros(num_bins, dtype=torch.float32)
    nonempty = counts > 0
    centers[nonempty] = sums[nonempty] / counts[nonempty]
    if (~nonempty).any():
        padded_edges = torch.cat([torch.tensor([-bound]), edges, torch.tensor([bound])])
        midpoints = (padded_edges[:-1] + padded_edges[1:]) / 2.0
        centers[~nonempty] = midpoints[~nonempty]
    return centers


def estimate_quantile_bins(
    all_values: torch.Tensor,
    bound: float,
    num_bins: int,
    lloyd_max_iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit non-uniform bin edges/centers to calibration data, clipped to [-bound, bound].

    This is a 1D Lloyd-Max quantizer: initialize edges uniformly over
    [-bound, bound], then iterate nearest-centroid reassignment (edges become
    midpoints between adjacent bin centroids) until convergence. Each iteration
    is guaranteed not to increase in-sample MSE, but the fixed point reached is
    only a local optimum — empirically, initializing from equiprobable quantile
    edges instead of uniform edges converges to a *worse* local optimum on this
    data (measured: ~10% higher MSE), so uniform init is used despite being a
    less "natural" starting point for a quantile-based scheme.
    """
    clipped = all_values.clamp(-bound, bound)
    edges = torch.linspace(-bound, bound, num_bins + 1)[1:-1]

    for _ in range(lloyd_max_iters):
        centers = _centroids_for_edges(clipped, edges, bound, num_bins)
        new_edges = (centers[:-1] + centers[1:]) / 2.0
        new_edges, _ = torch.sort(new_edges)
        if torch.allclose(new_edges, edges, atol=1e-6):
            edges = new_edges
            break
        edges = new_edges

    bin_centers = _centroids_for_edges(clipped, edges, bound, num_bins)
    return edges, bin_centers


def main() -> None:
    args = parse_args()
    dataset = CachedMicroLatentDataset(args.latent_cache)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    bases_path = Path(args.bases)
    bases = load_or_make_bases(bases_path, args.num_patches, args.patch_dim, args.seed)
    projector = RAMAProjector(bases).to(args.device)
    projector.requires_grad_(False)
    all_values = collect_projected_values(
        dataloader=dataloader,
        projector=projector,
        patch_size=args.patch_size,
        max_batches=args.max_batches,
        device=args.device,
    )
    bound = estimate_bound(all_values, args.percentile)
    config = {
        "num_bins": args.num_bins,
        "bound": bound,
        "bound_method": "percentile_abs_y",
        "percentile": args.percentile,
        "patch_size": args.patch_size,
        "patch_dim": args.patch_dim,
        "num_patches": args.num_patches,
        "mode": args.mode,
    }
    if args.mode == "quantile":
        edges, bin_centers = estimate_quantile_bins(all_values, bound, args.num_bins)
        config["edges"] = edges
        config["bin_centers"] = bin_centers
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(config, output)
    print(f"saved tokenizer config to {output}")
    print(f"mode={args.mode} bound={bound:.6f} percentile={args.percentile} num_bins={args.num_bins}")


if __name__ == "__main__":
    main()

