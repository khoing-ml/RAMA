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
from src.modules.rama import patchify
from src.rama.projector import RAMAProjector
from src.rama.tokenizer import RAMATokenizer, build_tokenizer_from_config, load_tokenizer_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect RAMA coordinates y before/after quantization binning.")
    parser.add_argument("--latent-cache", default="data/latents")
    parser.add_argument("--bases", default="cache/rama_bases_p256_d16.pt")
    parser.add_argument("--tokenizer-config", default="cache/rama_tokenizer_config.pt")
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--plot-out", default="outputs/quantization_tests/y_binning.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    dataset = CachedMicroLatentDataset(args.latent_cache)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    bases = torch.load(args.bases, map_location="cpu").float()
    projector = RAMAProjector(bases).to(args.device)

    tokenizer_config_path = Path(args.tokenizer_config)
    tokenizer = (
        build_tokenizer_from_config(load_tokenizer_config(str(tokenizer_config_path)))
        if tokenizer_config_path.exists()
        else RAMATokenizer()
    ).to(args.device)
    mode = "quantile" if tokenizer.quantile_based else "uniform"
    print(f"tokenizer: mode={mode} num_bins={tokenizer.num_bins} bound={tokenizer.bound:.4f}")

    y_all: list[torch.Tensor] = []
    y_hat_all: list[torch.Tensor] = []
    tokens_all: list[torch.Tensor] = []
    for step, batch in enumerate(dataloader):
        if step >= args.max_batches:
            break
        z_h = batch["z_H"].to(args.device)
        patches = patchify(z_h, patch_size=args.patch_size)
        y = projector.project(patches)
        tokens = tokenizer.quantize(y)
        y_hat = tokenizer.dequantize(tokens)
        y_all.append(y.cpu())
        y_hat_all.append(y_hat.cpu())
        tokens_all.append(tokens.cpu())

    y = torch.cat(y_all, dim=0)
    y_hat = torch.cat(y_hat_all, dim=0)
    tokens = torch.cat(tokens_all, dim=0)

    clip_frac = (y.abs() > tokenizer.bound).float().mean().item()
    err = (y_hat - y)
    mse = err.pow(2).mean().item()

    y_centered = y - y.mean()
    y_var = y_centered.pow(2).mean()
    excess_kurtosis = (y_centered.pow(4).mean() / y_var.pow(2) - 3.0).item()
    skewness = (y_centered.pow(3).mean() / y_var.pow(1.5)).item()

    print(f"\ny  : shape={tuple(y.shape)} mean={y.mean():.4f} std={y.std():.4f} min={y.min():.4f} max={y.max():.4f}")
    print(f"excess kurtosis: {excess_kurtosis:.3f}  (0=Gaussian, ~3=Laplace, higher=heavier tails)")
    print(f"skewness: {skewness:.3f}  (0=symmetric)")
    print(f"clip fraction (|y| > bound): {clip_frac:.4%}")
    if tokenizer.quantile_based:
        print(f"quantization MSE: {mse:.6f}")
    else:
        max_bin_err = (2.0 * tokenizer.bound / tokenizer.num_bins) / 2.0
        print(f"quantization MSE: {mse:.6f}  (ideal max per-value err at bin center = {max_bin_err:.6f})")

    # per-dimension spread, to check whether one global bound is appropriate across dims
    per_dim_std = y.std(dim=(0, 1))
    print(f"\nper-dim std of y: min={per_dim_std.min():.4f} max={per_dim_std.max():.4f} "
          f"ratio(max/min)={(per_dim_std.max() / per_dim_std.min()):.2f}")

    # bin occupancy: how evenly tokens use the available bins
    counts = torch.bincount(tokens.flatten(), minlength=tokenizer.num_bins).float()
    probs = counts / counts.sum()
    entropy = -(probs[probs > 0] * probs[probs > 0].log2()).sum().item()
    max_entropy = torch.log2(torch.tensor(float(tokenizer.num_bins))).item()
    empty_bins = (counts == 0).sum().item()
    print(f"\nbin occupancy: entropy={entropy:.2f} / max={max_entropy:.2f} bits  "
          f"({entropy / max_entropy:.1%} of ideal)  empty_bins={empty_bins}/{tokenizer.num_bins}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(y.flatten().numpy(), bins=200, alpha=0.6, label="y (pre-bin)")
        axes[0].hist(y_hat.flatten().numpy(), bins=200, alpha=0.6, label="y_hat (post-bin)")
        axes[0].axvline(-tokenizer.bound, color="red", linestyle="--", linewidth=1)
        axes[0].axvline(tokenizer.bound, color="red", linestyle="--", linewidth=1)
        axes[0].set_title("y distribution before/after binning")
        axes[0].legend()

        axes[1].bar(range(tokenizer.num_bins), counts.numpy())
        axes[1].set_title("token bin occupancy")
        axes[1].set_xlabel("bin index")

        axes[2].bar(range(per_dim_std.shape[0]), per_dim_std.numpy())
        axes[2].axhline(tokenizer.bound, color="red", linestyle="--", linewidth=1, label="bound")
        axes[2].set_title("per-dim std(y) vs global bound")
        axes[2].legend()

        out = Path(args.plot_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"\nsaved plot to {out}")
    except ImportError:
        print("\n(matplotlib not installed; skipping plot — numeric stats above are still valid)")


if __name__ == "__main__":
    main()
