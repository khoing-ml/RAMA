from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.modules.latent_decomposition import decompose_latent


class CachedLatentDataset(Dataset):
    """Loads cached SD-VAE latent tensors and returns macro latents z_L."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.paths = sorted(self.root.rglob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"no .pt latent files found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        item = torch.load(self.paths[index], map_location="cpu")
        if isinstance(item, torch.Tensor):
            z = item
            z_l = decompose_latent(z.unsqueeze(0) if z.ndim == 3 else z).z_l
        elif isinstance(item, dict):
            if "z_L" in item:
                z_l = item["z_L"]
            elif "z_l" in item:
                z_l = item["z_l"]
            elif "z" in item:
                z = item["z"]
                z_l = decompose_latent(z.unsqueeze(0) if z.ndim == 3 else z).z_l
            else:
                raise KeyError(f"{self.paths[index]} must contain z, z_L, or z_l")
        else:
            raise TypeError(f"unsupported latent cache item type: {type(item)!r}")

        if z_l.ndim == 4 and z_l.shape[0] == 1:
            z_l = z_l.squeeze(0)
        if z_l.ndim != 3:
            raise ValueError(f"expected z_L shape [C, H, W], got {tuple(z_l.shape)} from {self.paths[index]}")
        return z_l.float()
