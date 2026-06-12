from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.latent_dataset import CachedMicroLatentDataset
from src.data.latent_decomposition import reconstruct_from_decomposition
from src.data.vae import decode_latents, load_sd_vae
from src.evaluation.fid import InceptionFID, calculate_fid, stats_from_feature_batches
from src.macro.sampler import sample_macro_latents
from scripts.eval._common import collect_real_fid_stats, load_fid_stats, load_macro_model, resolve_vae_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate macro flow model FID: generated z_l + real z_h → full image."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to macro flow checkpoint .pt")
    parser.add_argument("--latents", required=True, help="Path to cached latent directory")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampler", choices=("heun", "euler", "shortcut"), default="heun")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--vae-checkpoint", default=None)
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--real-stats", default=None, help="Path to precomputed real FID stats (.pt)")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("Loading macro model...")
    model, config = load_macro_model(args.checkpoint, device, args.use_ema)
    macro_cfg = config.get("macro_flow_model", {})
    latent_channels = int(macro_cfg.get("in_channels", 4))
    latent_resolution = int(macro_cfg.get("resolution", 16))

    print("Loading dataset...")
    dataset = CachedMicroLatentDataset(args.latents)

    vae_ckpt = resolve_vae_checkpoint(config.get("vae", {}), args.vae_checkpoint)
    print(f"Loading VAE from {vae_ckpt}...")
    vae = load_sd_vae(vae_ckpt, cache_dir=args.cache_dir, dtype=args.dtype, device=str(device))

    fid_model = InceptionFID(device)

    if args.real_stats:
        print(f"Loading precomputed real FID stats from {args.real_stats}...")
        real_stats = load_fid_stats(args.real_stats)
        print(f"  Loaded stats for {real_stats.num_samples} samples.")
    else:
        print(f"Computing real FID stats ({min(args.num_samples, len(dataset))} samples)...")
        real_stats = collect_real_fid_stats(dataset, vae, fid_model, args.num_samples, args.batch_size, device)

    print("Generating fake images (generated z_l + real z_h)...")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    fake_batches: list[torch.Tensor] = []
    remaining = args.num_samples
    for batch in loader:
        if remaining <= 0:
            break
        current = min(remaining, batch["z_H"].shape[0])
        z_h_real = batch["z_H"][:current].to(device)
        shape = (current, latent_channels, latent_resolution, latent_resolution)
        z_l_fake = sample_macro_latents(model, shape=shape, method=args.sampler, num_steps=args.steps, device=device)
        z_fake = reconstruct_from_decomposition(z_l_fake, z_h_real)
        fake_batches.append(fid_model(decode_latents(vae, z_fake)))
        remaining -= current

    fid = calculate_fid(real_stats, stats_from_feature_batches(fake_batches))
    print(f"Macro FID: {fid:.4f}")


if __name__ == "__main__":
    main()
