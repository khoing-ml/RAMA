from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.latent_dataset import CachedMicroLatentDataset
from src.dataset.vae import load_sd_vae
from src.evaluation.fid import InceptionFID
from scripts.eval._common import collect_real_fid_stats, save_fid_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute real-data FID statistics from cached latents and save to disk."
    )
    parser.add_argument("--latents", required=True, help="Path to cached latent directory")
    parser.add_argument("--output", required=True, help="Output path for saved FID stats (.pt)")
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vae-checkpoint", default=None, help="VAE checkpoint path or HF model id")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("Loading dataset...")
    dataset = CachedMicroLatentDataset(args.latents)
    num_samples = min(args.num_samples, len(dataset))
    print(f"  {len(dataset)} samples found, using {num_samples}")

    vae_ckpt = args.vae_checkpoint or "stabilityai/sd-vae-ft-mse"
    print(f"Loading VAE from {vae_ckpt}...")
    vae = load_sd_vae(vae_ckpt, cache_dir=args.cache_dir, dtype=args.dtype, device=str(device))

    print("Loading Inception-V3 FID model...")
    fid_model = InceptionFID(device)

    print(f"Computing real FID stats ({num_samples} samples)...")
    stats = collect_real_fid_stats(dataset, vae, fid_model, num_samples, args.batch_size, device)

    save_fid_stats(stats, args.output)
    print(f"Saved FID stats ({stats.num_samples} samples) to {args.output}")


if __name__ == "__main__":
    main()
