from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.transforms import functional as TF


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def load_sd_vae(
    checkpoint: str = "stabilityai/sd-vae-ft-mse",
    cache_dir: str | None = ".cache/huggingface",
    dtype: str = "fp16",
    device: str = "cuda",
) -> AutoencoderKL:
    torch_dtype = resolve_dtype(dtype)
    if checkpoint.endswith((".ckpt", ".safetensors")):
        vae = AutoencoderKL.from_single_file(checkpoint, torch_dtype=torch_dtype)
    else:
        vae = AutoencoderKL.from_pretrained(
            checkpoint,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
        )
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def load_image_tensor(path: str | Path, image_size: int = 256) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = TF.resize(image, image_size, interpolation=TF.InterpolationMode.BICUBIC)
    image = TF.center_crop(image, [image_size, image_size])
    tensor = TF.to_tensor(image)
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    return TF.to_pil_image(tensor.squeeze(0))


@torch.no_grad()
def encode_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    images = images.to(device=vae.device, dtype=vae.dtype)
    posterior = vae.encode(images).latent_dist
    z = posterior.sample()
    return z * vae.config.scaling_factor


@torch.no_grad()
def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.to(device=vae.device, dtype=vae.dtype)
    return vae.decode(latents / vae.config.scaling_factor).sample

