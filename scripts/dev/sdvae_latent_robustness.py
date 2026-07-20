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

from src.dataset.vae import decode_latents, encode_latents, load_sd_vae

# severities expressed as a fraction of each latent's own per-image std
NOISE_LEVELS = [0.10, 0.25, 0.50, 0.75, 1.00]


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_dtype = "fp16" if default_device == "cuda" else "fp32"

    parser = argparse.ArgumentParser(description="Test SD-VAE decoder robustness to additive latent-space noise.")
    parser.add_argument("--data-dir", default="data/celeba256")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/sdvae_latent_robustness")
    parser.add_argument("--checkpoint", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--dtype", default=default_dtype, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--noise-seed", type=int, default=42, help="Seed for the latent noise itself (kept fixed across images so severities are comparable).")
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


def save_grid(rows: list[list[np.ndarray]], row_labels: list[str], out_path: Path) -> None:
    from PIL import ImageDraw

    cell_h, cell_w = rows[0][0].shape[0], rows[0][0].shape[1]
    label_w = 140
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
    print(f"testing latent-noise robustness on {len(sample_paths)} images from {args.data_dir}")

    noise_gen = torch.Generator(device="cpu").manual_seed(args.noise_seed)

    results = []
    for img_idx, path in enumerate(sample_paths):
        image = Image.open(path).convert("RGB").resize((args.image_size, args.image_size), Image.BICUBIC)
        clean = np.array(image)

        clean_tensor = image_to_tensor(clean)
        z_clean = encode_latents(vae, clean_tensor)
        recon_clean = to_uint8(decode_latents(vae, z_clean))
        clean_metrics = compute_metrics(clean, recon_clean)
        results.append({"image": path.name, "noise_level": 0.0, "recon_vs_clean": clean_metrics})

        z_std = z_clean.float().std().item()

        grid_rows = [[clean, recon_clean]]
        row_labels = ["clean"]

        for level in NOISE_LEVELS:
            sigma = level * z_std
            noise = torch.randn(z_clean.shape, generator=noise_gen).to(z_clean.device, z_clean.dtype) * sigma
            z_noisy = z_clean + noise
            recon_noisy = to_uint8(decode_latents(vae, z_noisy))
            metrics = compute_metrics(clean, recon_noisy)
            results.append({"image": path.name, "noise_level": level, "recon_vs_clean": metrics})

            if img_idx == 0:
                grid_rows.append([recon_noisy])
                row_labels.append(f"z-noise {level:.2f}×σ")

        if img_idx == 0:
            # pad clean row to 1 column to match single-column noisy rows for the grid helper
            save_grid([[clean], [recon_clean]] + grid_rows[1:], ["clean input", "clean recon"] + row_labels[1:], grid_dir / "example_grid.png")

        print(f"[{img_idx + 1}/{len(sample_paths)}] {path.name} done (latent std={z_std:.3f})")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} records to {out_dir / 'results.json'}")
    print(f"wrote example grid to {grid_dir / 'example_grid.png'}")


if __name__ == "__main__":
    main()
