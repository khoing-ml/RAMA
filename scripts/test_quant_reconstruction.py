from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.latent_dataset import CachedMicroLatentDataset
from src.modules.latent_decomposition import decompose_latent
from src.modules.rama import patchify, unpatchify
from src.modules.vae_utils import decode_latents, load_sd_vae
from src.rama.projector import RAMAProjector
from src.rama.tokenizer import RAMATokenizer, build_tokenizer_from_config, load_tokenizer_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantization-only residual reconstruction.")
    parser.add_argument("--latent-cache", default="data/latents")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--bases", default="cache/rama_bases_p256_d16.pt")
    parser.add_argument("--tokenizer-config", default="cache/rama_tokenizer_config.pt")
    parser.add_argument("--out", default="outputs/quantization_tests/vae_vs_macro_vs_quant.png")
    parser.add_argument("--checkpoint", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch-size", type=int, default=2)
    return parser.parse_args()


@torch.no_grad()
def quantization_only_reconstruct(
    z: torch.Tensor,
    projector: RAMAProjector,
    tokenizer: RAMATokenizer,
    patch_size: int = 2,
) -> torch.Tensor:
    decomposition = decompose_latent(z)
    patches = patchify(decomposition.z_h, patch_size=patch_size)
    y = projector.project(patches)
    tokens = tokenizer.quantize(y)
    y_hat = tokenizer.dequantize(tokens)
    patches_hat = projector.inverse(y_hat)
    z_h_hat = unpatchify(
        patches_hat,
        channels=z.shape[1],
        height=z.shape[2],
        width=z.shape[3],
        patch_size=patch_size,
    )
    z_l_up = F.interpolate(decomposition.z_l, size=z.shape[-2:], mode="bilinear", align_corners=False)
    return z_l_up + z_h_hat


def main() -> None:
    args = parse_args()
    dataset = CachedMicroLatentDataset(args.latent_cache)
    item = dataset[args.index]
    if "z" in item:
        z = item["z"].unsqueeze(0)
        decomposition = decompose_latent(z)
    else:
        z_l = item["z_L"].unsqueeze(0)
        z_h = item["z_H"].unsqueeze(0)
        z_l_up = F.interpolate(z_l, size=z_h.shape[-2:], mode="bilinear", align_corners=False)
        z = z_l_up + z_h
        decomposition = decompose_latent(z)

    bases = torch.load(args.bases, map_location="cpu").float()
    projector = RAMAProjector(bases).to(args.device)
    tokenizer_config = Path(args.tokenizer_config)
    tokenizer = (
        build_tokenizer_from_config(load_tokenizer_config(str(tokenizer_config)))
        if tokenizer_config.exists()
        else RAMATokenizer()
    )

    z = z.to(args.device)
    decomposition = decompose_latent(z)
    z_quant = quantization_only_reconstruct(z, projector, tokenizer, patch_size=args.patch_size)
    vae = load_sd_vae(checkpoint=args.checkpoint, cache_dir=args.cache_dir, dtype=args.dtype, device=args.device)
    images = torch.cat(
        [
            decode_latents(vae, z),
            decode_latents(vae, decomposition.z_l_up),
            decode_latents(vae, z_quant),
        ],
        dim=0,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((images.float().clamp(-1, 1) + 1.0) / 2.0, out, nrow=3)
    latent_mse = F.mse_loss(z_quant.float(), z.float()).item()
    print(f"saved {out}")
    print(f"quantized latent MSE: {latent_mse:.8f}")


if __name__ == "__main__":
    main()

