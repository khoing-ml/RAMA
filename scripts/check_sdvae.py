from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.latent_decomposition import decompose_latent, reconstruct_from_decomposition
from src.modules.vae_utils import decode_latents, encode_latents, load_image_tensor, load_sd_vae, tensor_to_image


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_dtype = "fp16" if default_device == "cuda" else "fp32"

    parser = argparse.ArgumentParser(description="Run SD-VAE latent decomposition sanity checks.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--out", default="outputs/sdvae_check", help="Output directory.")
    parser.add_argument("--checkpoint", default="stabilityai/sd-vae-ft-mse", help="HF repo ID, local Diffusers folder, or single VAE checkpoint file.")
    parser.add_argument("--cache-dir", default=".cache/huggingface", help="Hugging Face cache directory.")
    parser.add_argument("--dtype", default=default_dtype, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae = load_sd_vae(
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        device=args.device,
    )
    images = load_image_tensor(args.image, image_size=args.image_size)

    z = encode_latents(vae, images)
    decomposition = decompose_latent(z)

    vae_reconstruction = decode_latents(vae, z)
    macro_reconstruction = decode_latents(vae, decomposition.z_l_up)
    zh_reconstruction = decode_latents(vae, decomposition.z_h)
    full_reconstruction = decode_latents(
        vae,
        reconstruct_from_decomposition(decomposition.z_l, decomposition.z_h),
    )

    tensor_to_image(vae_reconstruction).save(out_dir / "vae_reconstruction.png")
    tensor_to_image(macro_reconstruction).save(out_dir / "macro_reconstruction.png")
    tensor_to_image(zh_reconstruction).save(out_dir / "zh_reconstruction.png")
    tensor_to_image(full_reconstruction).save(out_dir / "full_decomposition_reconstruction.png")

    torch.save(
        {
            "z": z.detach().cpu().half(),
            "z_L": decomposition.z_l.detach().cpu().half(),
            "z_H": decomposition.z_h.detach().cpu().half(),
            "scaling_factor": vae.config.scaling_factor,
        },
        out_dir / "latents.pt",
    )

    max_error = (z - reconstruct_from_decomposition(decomposition.z_l, decomposition.z_h)).abs().max().item()
    print(f"wrote outputs to {out_dir}")
    print(f"z shape: {tuple(z.shape)}")
    print(f"z_L shape: {tuple(decomposition.z_l.shape)}")
    print(f"z_H shape: {tuple(decomposition.z_h.shape)}")
    print(f"full decomposition max latent error: {max_error:.8f}")


if __name__ == "__main__":
    main()
