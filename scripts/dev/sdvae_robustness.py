from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.vae import decode_latents, encode_latents, load_sd_vae, tensor_to_image


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_dtype = "fp16" if default_device == "cuda" else "fp32"

    parser = argparse.ArgumentParser(description="Test SD-VAE reconstruction robustness under input corruptions.")
    parser.add_argument("--data-dir", default="data/celeba256", help="Directory of source images.")
    parser.add_argument("--num-images", type=int, default=8, help="Number of images to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/sdvae_robustness", help="Output directory.")
    parser.add_argument("--checkpoint", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--dtype", default=default_dtype, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--severities", type=int, default=5, help="Number of severity levels (1..N).")
    return parser.parse_args()


# --- corruptions, operating on uint8 HWC numpy arrays in [0, 255] ---

def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    sigma = [0.04, 0.08, 0.12, 0.18, 0.26][severity - 1] * 255.0
    x = img.astype(np.float32) + np.random.normal(0, sigma, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, severity: int) -> np.ndarray:
    radius = [0.5, 1.0, 2.0, 3.5, 5.0][severity - 1]
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius)))


def jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    quality = [60, 40, 25, 15, 5][severity - 1]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def brightness(img: np.ndarray, severity: int) -> np.ndarray:
    delta = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1] * 255.0
    x = img.astype(np.float32) + delta
    return np.clip(x, 0, 255).astype(np.uint8)


def contrast(img: np.ndarray, severity: int) -> np.ndarray:
    factor = [0.8, 0.6, 0.4, 0.25, 0.1][severity - 1]
    mean = img.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    x = (img.astype(np.float32) - mean) * factor + mean
    return np.clip(x, 0, 255).astype(np.uint8)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "jpeg_compression": jpeg_compression,
    "brightness": brightness,
    "contrast": contrast,
}


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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.out)
    grid_dir = out_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    vae = load_sd_vae(checkpoint=args.checkpoint, cache_dir=args.cache_dir, dtype=args.dtype, device=args.device)

    all_paths = sorted(Path(args.data_dir).glob("*.jpg")) + sorted(Path(args.data_dir).glob("*.png"))
    sample_paths = random.sample(all_paths, min(args.num_images, len(all_paths)))
    print(f"testing on {len(sample_paths)} images from {args.data_dir}")

    results = []
    for img_idx, path in enumerate(sample_paths):
        image = Image.open(path).convert("RGB").resize((args.image_size, args.image_size), Image.BICUBIC)
        clean = np.array(image)

        clean_tensor = image_to_tensor(clean)
        z_clean = encode_latents(vae, clean_tensor)
        recon_clean = to_uint8(decode_latents(vae, z_clean))
        clean_recon_metrics = compute_metrics(clean, recon_clean)
        results.append({
            "image": path.name, "corruption": "none", "severity": 0,
            "input_vs_clean": {"psnr": float("inf"), "ssim": 1.0},
            "recon_vs_clean": clean_recon_metrics,
        })

        grid_rows = [[clean, recon_clean]]
        row_labels = ["clean"]

        for corruption_name, fn in CORRUPTIONS.items():
            severity_row = []
            for severity in range(1, args.severities + 1):
                corrupted = fn(clean, severity)
                corrupted_tensor = image_to_tensor(corrupted)
                z_corrupted = encode_latents(vae, corrupted_tensor)
                recon_corrupted = to_uint8(decode_latents(vae, z_corrupted))

                results.append({
                    "image": path.name,
                    "corruption": corruption_name,
                    "severity": severity,
                    "input_vs_clean": compute_metrics(clean, corrupted),
                    "recon_vs_clean": compute_metrics(clean, recon_corrupted),
                })

                if img_idx == 0:
                    severity_row.append((corruption_name, severity, corrupted, recon_corrupted))

            if img_idx == 0:
                for corruption_name, severity, corrupted, recon_corrupted in severity_row:
                    grid_rows.append([corrupted, recon_corrupted])
                    row_labels.append(f"{corruption_name} s{severity}")

        if img_idx == 0:
            save_grid(grid_rows, row_labels, grid_dir / "example_grid.png")

        print(f"[{img_idx + 1}/{len(sample_paths)}] {path.name} done")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} records to {out_dir / 'results.json'}")
    print(f"wrote example grid to {grid_dir / 'example_grid.png'}")


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


if __name__ == "__main__":
    main()
