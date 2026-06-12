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
from src.macro.sampler import sample_macro_latents
from src.modules.micro_rama import sample_micro_latent
from src.modules.rama import unpatchify
from src.rama.projector import RAMAProjector
from src.rama.tokenizer import build_tokenizer_from_config, load_tokenizer_config
from scripts.eval._common import (
    collect_real_fid_stats,
    load_macro_model,
    load_micro_models,
    load_or_make_bases,
    resolve_micro_type,
    resolve_patch_size,
    resolve_vae_checkpoint,
    sample_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end FID: macro generates z_l, micro generates z_h, combine and evaluate."
    )
    parser.add_argument("--macro-checkpoint", required=True, help="Path to macro flow checkpoint .pt")
    parser.add_argument("--micro-checkpoint", required=True, help="Path to micro RAMA checkpoint .pt")
    parser.add_argument("--latents", required=True, help="Path to cached latent directory (for real FID stats)")
    parser.add_argument("--bases", default=None, help="Path to RAMA bases .pt (default: from config or cache/)")
    parser.add_argument("--tokenizer-config", default=None, help="Path to tokenizer config .pt (categorical only)")
    parser.add_argument("--micro-type", choices=("auto", "categorical", "continuous"), default="auto")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampler", choices=("heun", "euler", "shortcut"), default="heun")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use-argmax", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Gaussian base-noise scale (continuous only)")
    parser.add_argument("--vae-checkpoint", default=None)
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("Loading macro model...")
    macro_model, macro_config = load_macro_model(args.macro_checkpoint, device, args.use_ema)
    macro_cfg = macro_config.get("macro_flow_model", {})
    latent_channels = int(macro_cfg.get("in_channels", 4))
    latent_resolution = int(macro_cfg.get("resolution", 16))

    # Resolve micro_type before building micro models
    micro_ckpt_config = torch.load(args.micro_checkpoint, map_location="cpu").get("config", {})
    micro_type = resolve_micro_type(micro_ckpt_config, args.micro_type)

    tokenizer = None
    if micro_type == "categorical":
        tok_path = args.tokenizer_config or str(micro_ckpt_config.get("tokenizer", {}).get("config_path", "cache/rama_tokenizer_config.pt"))
        tokenizer = build_tokenizer_from_config(load_tokenizer_config(tok_path))

    print(f"Loading micro models (type={micro_type})...")
    context_encoder, micro_model, micro_config = load_micro_models(args.micro_checkpoint, micro_type, tokenizer, device)

    rama_cfg = micro_config.get("rama", micro_config.get("rama_bases", {}))
    bases = load_or_make_bases(rama_cfg, args.bases).to(device)
    projector = RAMAProjector(bases)
    patch_size = resolve_patch_size(micro_config)

    # Resolve residual latent shape from micro config
    residual_shape = micro_config.get("micro_latent", {}).get("residual_shape", None)
    if residual_shape is not None:
        res_channels, res_height, res_width = int(residual_shape[0]), int(residual_shape[1]), int(residual_shape[2])
    else:
        res_channels = latent_channels
        res_height = res_width = latent_resolution * 2  # default: macro is 2x downsampled

    print("Loading dataset (for real FID stats)...")
    dataset = CachedMicroLatentDataset(args.latents)

    vae_ckpt = resolve_vae_checkpoint(macro_config.get("vae", {}), args.vae_checkpoint)
    print(f"Loading VAE from {vae_ckpt}...")
    vae = load_sd_vae(vae_ckpt, cache_dir=args.cache_dir, dtype=args.dtype, device=str(device))

    fid_model = InceptionFID(device)

    print(f"Computing real FID stats ({min(args.num_samples, len(dataset))} samples)...")
    real_stats = collect_real_fid_stats(dataset, vae, fid_model, args.num_samples, args.batch_size, device)

    print("Generating fake images (macro z_l → micro z_h → full)...")
    fake_batches: list[torch.Tensor] = []
    remaining = args.num_samples
    while remaining > 0:
        current = min(remaining, args.batch_size)
        shape = (current, latent_channels, latent_resolution, latent_resolution)
        z_l = sample_macro_latents(macro_model, shape=shape, method=args.sampler, num_steps=args.steps, device=device)

        if micro_type == "categorical":
            context = context_encoder(z_l)
            logits = micro_model(context)
            tokens = sample_tokens(logits, temperature=args.temperature, use_argmax=args.use_argmax)
            y_hat = tokenizer.dequantize(tokens)
            patches_hat = projector.inverse(y_hat)
            z_h_hat = unpatchify(patches_hat, channels=res_channels, height=res_height, width=res_width, patch_size=patch_size)
        else:
            z_h_hat = sample_micro_latent(
                z_l, context_encoder, micro_model, projector.bases,
                latent_channels=res_channels, latent_height=res_height, latent_width=res_width,
                patch_size=patch_size, noise_scale=args.noise_scale,
            )

        z_fake = reconstruct_from_decomposition(z_l, z_h_hat)
        fake_batches.append(fid_model(decode_latents(vae, z_fake)))
        remaining -= current

    fid = calculate_fid(real_stats, stats_from_feature_batches(fake_batches))
    print(f"Full (end-to-end) FID: {fid:.4f}")


if __name__ == "__main__":
    main()
