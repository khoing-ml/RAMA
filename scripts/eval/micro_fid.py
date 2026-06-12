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
from src.dataset.latent_decomposition import reconstruct_from_decomposition
from src.dataset.vae import decode_latents, load_sd_vae
from src.evaluation.fid import InceptionFID, calculate_fid, stats_from_feature_batches
from src.modules.micro_rama import sample_micro_latent
from src.modules.rama import unpatchify
from src.rama.projector import RAMAProjector
from src.rama.tokenizer import build_tokenizer_from_config, load_tokenizer_config
from scripts.eval._common import (
    collect_real_fid_stats,
    load_fid_stats,
    load_micro_models,
    load_or_make_bases,
    resolve_micro_type,
    resolve_patch_size,
    resolve_vae_checkpoint,
    sample_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate micro RAMA model FID: real z_l → generated z_h → full image."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to micro RAMA checkpoint .pt")
    parser.add_argument("--latents", required=True, help="Path to cached latent directory")
    parser.add_argument("--bases", default=None, help="Path to RAMA bases .pt (default: from config or cache/)")
    parser.add_argument("--tokenizer-config", default=None, help="Path to tokenizer config .pt (categorical only)")
    parser.add_argument("--micro-type", choices=("auto", "categorical", "continuous"), default="auto")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use-argmax", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Gaussian base-noise scale (continuous only)")
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

    # Load a temporary checkpoint to resolve micro_type before building models
    ckpt_config = torch.load(args.checkpoint, map_location="cpu").get("config", {})
    micro_type = resolve_micro_type(ckpt_config, args.micro_type)

    tokenizer = None
    if micro_type == "categorical":
        tok_path = args.tokenizer_config or str(ckpt_config.get("tokenizer", {}).get("config_path", "cache/rama_tokenizer_config.pt"))
        tokenizer = build_tokenizer_from_config(load_tokenizer_config(tok_path))

    print(f"Loading micro models (type={micro_type})...")
    context_encoder, micro_model, config = load_micro_models(args.checkpoint, micro_type, tokenizer, device)

    rama_cfg = config.get("rama", config.get("rama_bases", {}))
    bases = load_or_make_bases(rama_cfg, args.bases).to(device)
    projector = RAMAProjector(bases)
    patch_size = resolve_patch_size(config)

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

    print("Generating fake images (real z_l → generated z_h)...")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    fake_batches: list[torch.Tensor] = []
    remaining = args.num_samples
    for batch in loader:
        if remaining <= 0:
            break
        z_l = batch["z_L"][:remaining].to(device)
        z_h = batch["z_H"][:remaining].to(device)

        if micro_type == "categorical":
            context = context_encoder(z_l)
            logits = micro_model(context)
            tokens = sample_tokens(logits, temperature=args.temperature, use_argmax=args.use_argmax)
            y_hat = tokenizer.dequantize(tokens)
            patches_hat = projector.inverse(y_hat)
            z_h_hat = unpatchify(patches_hat, channels=z_h.shape[1], height=z_h.shape[2], width=z_h.shape[3], patch_size=patch_size)
        else:
            z_h_hat = sample_micro_latent(
                z_l, context_encoder, micro_model, projector.bases,
                latent_channels=z_h.shape[1], latent_height=z_h.shape[2], latent_width=z_h.shape[3],
                patch_size=patch_size, noise_scale=args.noise_scale,
            )

        z_fake = reconstruct_from_decomposition(z_l, z_h_hat)
        fake_batches.append(fid_model(decode_latents(vae, z_fake)))
        remaining -= z_l.shape[0]

    fid = calculate_fid(real_stats, stats_from_feature_batches(fake_batches))
    print(f"Micro FID: {fid:.4f}")


if __name__ == "__main__":
    main()
