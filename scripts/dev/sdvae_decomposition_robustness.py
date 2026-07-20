from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.latent_decomposition import decompose_latent, reconstruct_from_decomposition
from src.dataset.vae import decode_latents, encode_latents, load_sd_vae

# severities expressed as a fraction of each component's own per-image std
NOISE_LEVELS = [0.10, 0.25, 0.50, 0.75, 1.00]


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_dtype = "fp16" if default_device == "cuda" else "fp32"

    parser = argparse.ArgumentParser(
        description="Decompose SD-VAE latents into low-freq (macro, z_L) and high-freq (micro, z_H) "
        "components, perturb each independently, merge back, and measure the effect on reconstruction."
    )
    parser.add_argument("--data-dir", default="data/celeba256")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/sdvae_decomposition_robustness")
    parser.add_argument("--checkpoint", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--dtype", default=default_dtype, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--noise-seed", type=int, default=42)
    return parser.parse_args()


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0)


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0 * 255.0
    return tensor.squeeze(0).permute(1, 2, 0).round().numpy().astype(np.uint8)


def compute_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "psnr": float(psnr(a, b, data_range=255)),
        "ssim": float(ssim(a, b, channel_axis=2, data_range=255)),
    }


def add_noise(tensor: torch.Tensor, level: float, gen: torch.Generator) -> torch.Tensor:
    sigma = level * tensor.float().std().item()
    noise = torch.randn(tensor.shape, generator=gen).to(tensor.device, tensor.dtype) * sigma
    return tensor + noise


def save_grid(rows: list[list[np.ndarray]], row_labels: list[str], out_path: Path) -> None:
    from PIL import ImageDraw

    cell_h, cell_w = rows[0][0].shape[0], rows[0][0].shape[1]
    label_w = 160
    n_cols = len(rows[0])
    grid = Image.new("RGB", (label_w + n_cols * cell_w, len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(grid)
    for r, (row, label) in enumerate(zip(rows, row_labels)):
        draw.text((8, r * cell_h + cell_h // 2 - 8), label, fill="black")
        for c, cell in enumerate(row):
            grid.paste(Image.fromarray(cell), (label_w + c * cell_w, r * cell_h))
    grid.save(out_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.out)
    grid_dir = out_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    vae = load_sd_vae(checkpoint=args.checkpoint, cache_dir=args.cache_dir, dtype=args.dtype, device=args.device)

    all_paths = sorted(Path(args.data_dir).glob("*.jpg")) + sorted(Path(args.data_dir).glob("*.png"))
    sample_paths = random.sample(all_paths, min(args.num_images, len(all_paths)))
    print(f"testing macro/micro decomposition robustness on {len(sample_paths)} images from {args.data_dir}")

    results = []
    for img_idx, path in enumerate(sample_paths):
        image = Image.open(path).convert("RGB").resize((args.image_size, args.image_size), Image.BICUBIC)
        clean = np.array(image)

        clean_tensor = image_to_tensor(clean)
        z = encode_latents(vae, clean_tensor)
        decomp = decompose_latent(z)
        z_l, z_h = decomp.z_l, decomp.z_h

        recon_clean = to_uint8(decode_latents(vae, reconstruct_from_decomposition(z_l, z_h)))
        clean_metrics = compute_metrics(clean, recon_clean)
        results.append({"image": path.name, "band": "none", "noise_level": 0.0, "recon_vs_clean": clean_metrics})

        macro_gen = torch.Generator(device="cpu").manual_seed(args.noise_seed)
        micro_gen = torch.Generator(device="cpu").manual_seed(args.noise_seed)

        macro_row = []
        micro_row = []
        for level in NOISE_LEVELS:
            z_l_noisy = add_noise(z_l, level, macro_gen)
            recon_macro = to_uint8(decode_latents(vae, reconstruct_from_decomposition(z_l_noisy, z_h)))
            results.append({
                "image": path.name, "band": "macro_zL", "noise_level": level,
                "recon_vs_clean": compute_metrics(clean, recon_macro),
            })

            z_h_noisy = add_noise(z_h, level, micro_gen)
            recon_micro = to_uint8(decode_latents(vae, reconstruct_from_decomposition(z_l, z_h_noisy)))
            results.append({
                "image": path.name, "band": "micro_zH", "noise_level": level,
                "recon_vs_clean": compute_metrics(clean, recon_micro),
            })

            if img_idx == 0:
                macro_row.append(recon_macro)
                micro_row.append(recon_micro)

        if img_idx == 0:
            rows = [[clean, recon_clean]] + [[m] for m in macro_row]
            labels = ["clean input | clean recon"] + [f"macro (z_L) noise {lv:.2f}σ" for lv in NOISE_LEVELS]
            save_grid(rows, labels, grid_dir / "macro_noise_grid.png")

            rows = [[clean, recon_clean]] + [[m] for m in micro_row]
            labels = ["clean input | clean recon"] + [f"micro (z_H) noise {lv:.2f}σ" for lv in NOISE_LEVELS]
            save_grid(rows, labels, grid_dir / "micro_noise_grid.png")

        print(f"[{img_idx + 1}/{len(sample_paths)}] {path.name} done "
              f"(z_L std={z_l.float().std().item():.3f}, z_H std={z_h.float().std().item():.3f})")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} records to {out_dir / 'results.json'}")
    print(f"wrote grids to {grid_dir}")


if __name__ == "__main__":
    main()
