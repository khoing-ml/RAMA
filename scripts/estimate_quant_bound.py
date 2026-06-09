from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.latent_dataset import CachedMicroLatentDataset
from src.modules.rama import make_orthogonal_bases, patchify
from src.rama.projector import RAMAProjector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate RAMA tokenizer clipping bound from cached latents.")
    parser.add_argument("--latent-cache", default="data/latents")
    parser.add_argument("--bases", default="cache/rama_bases_p256_d16.pt")
    parser.add_argument("--output", default="cache/rama_tokenizer_config.pt")
    parser.add_argument("--num-bins", type=int, default=256)
    parser.add_argument("--percentile", type=float, default=99.5)
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
def estimate_quant_bound(
    dataloader: DataLoader,
    projector: RAMAProjector,
    patch_size: int = 2,
    percentile: float = 99.5,
    max_batches: int = 200,
    device: str | torch.device = "cuda",
) -> float:
    values: list[torch.Tensor] = []
    for step, batch in enumerate(dataloader):
        if step >= max_batches:
            break
        z_h = batch["z_H"].to(device)
        patches = patchify(z_h, patch_size=patch_size)
        y = projector.project(patches)
        values.append(y.abs().flatten().cpu())
    if not values:
        raise ValueError("no values collected; check latent cache and max_batches")
    all_values = torch.cat(values, dim=0)
    return float(torch.quantile(all_values, percentile / 100.0).item())


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
    bound = estimate_quant_bound(
        dataloader=dataloader,
        projector=projector,
        patch_size=args.patch_size,
        percentile=args.percentile,
        max_batches=args.max_batches,
        device=args.device,
    )
    config = {
        "num_bins": args.num_bins,
        "bound": bound,
        "bound_method": "percentile_abs_y",
        "percentile": args.percentile,
        "patch_size": args.patch_size,
        "patch_dim": args.patch_dim,
        "num_patches": args.num_patches,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(config, output)
    print(f"saved tokenizer config to {output}")
    print(f"bound={bound:.6f} percentile={args.percentile} num_bins={args.num_bins}")


if __name__ == "__main__":
    main()

