from __future__ import annotations

import argparse
from io import BytesIO
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.latent_decomposition import decompose_latent
from src.modules.vae_utils import encode_latents, load_image_tensor, load_sd_vae


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImagePathDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.paths = sorted(path for path in self.root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise FileNotFoundError(f"no images found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        image = load_image_tensor(path, image_size=self.image_size).squeeze(0)
        return image, path.stem


class ParquetImageDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("install the datasets package to cache images from Parquet shards") from exc

        self.root = Path(root)
        self.image_size = image_size
        data_files = sorted(str(path) for path in self.root.rglob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"no parquet files found under {self.root}")
        self.dataset = load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            cache_dir=str(self.root / ".cache" / "datasets"),
        )
        self.image_column = self._resolve_image_column()

    def _resolve_image_column(self) -> str:
        for name in ("image", "img"):
            if name in self.dataset.column_names:
                return name
        sample = self.dataset[0]
        for name, value in sample.items():
            if isinstance(value, Image.Image):
                return name
            if isinstance(value, dict) and ("bytes" in value or "path" in value):
                return name
        raise KeyError(f"could not find an image column in parquet columns: {self.dataset.column_names}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        example = self.dataset[index]
        image = self._to_pil_image(example[self.image_column])
        image = TF.resize(image.convert("RGB"), self.image_size, interpolation=TF.InterpolationMode.BICUBIC)
        image = TF.center_crop(image, [self.image_size, self.image_size])
        tensor = TF.to_tensor(image) * 2.0 - 1.0
        stem = str(example.get("id", example.get("file_name", f"{index:06d}")))
        return tensor, Path(stem).stem

    @staticmethod
    def _to_pil_image(value: object) -> Image.Image:
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(BytesIO(value["bytes"]))
            if value.get("path") is not None:
                return Image.open(value["path"])
        raise TypeError(f"unsupported image value from parquet dataset: {type(value)!r}")


def build_image_dataset(root: str | Path, image_size: int) -> Dataset:
    root = Path(root)
    image_paths = [path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if image_paths:
        return ImagePathDataset(root, image_size=image_size)
    parquet_paths = list(root.rglob("*.parquet"))
    if parquet_paths:
        return ParquetImageDataset(root, image_size=image_size)
    raise FileNotFoundError(f"no images or parquet shards found under {root}")


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_dtype = "fp16" if default_device == "cuda" else "fp32"
    parser = argparse.ArgumentParser(description="Cache SD-VAE latents for macro flow training.")
    parser.add_argument("--images", default="data/celeba256", help="Directory of input images.")
    parser.add_argument("--out", default="data/latents", help="Output latent cache directory.")
    parser.add_argument("--checkpoint", default="stabilityai/sd-vae-ft-mse", help="HF repo ID, local Diffusers folder, or single VAE checkpoint file.")
    parser.add_argument("--cache-dir", default=".cache/huggingface", help="Hugging Face cache directory.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--dtype", default=default_dtype, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--store-components", action="store_true", help="Also store z_L and z_H in each cache file.")
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
    dataset = build_image_dataset(args.images, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    written = 0
    for images, stems in tqdm(loader, desc="caching latents"):
        z = encode_latents(vae, images)
        decomposition = decompose_latent(z)
        for i, stem in enumerate(stems):
            item = {
                "z": z[i].detach().cpu().half(),
                "scaling_factor": vae.config.scaling_factor,
            }
            if args.store_components:
                item["z_L"] = decomposition.z_l[i].detach().cpu().half()
                item["z_H"] = decomposition.z_h[i].detach().cpu().half()
            torch.save(item, out_dir / f"{written:06d}_{stem}.pt")
            written += 1

    print(f"wrote {written} latent files to {out_dir}")


if __name__ == "__main__":
    main()
