from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.dataset.latent_decomposition import decompose_latent


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


class CachedMicroLatentDataset(Dataset):
    """Loads cached latents and returns full, macro, and residual latents."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.paths = sorted(self.root.rglob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"no .pt latent files found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = torch.load(self.paths[index], map_location="cpu")
        if isinstance(item, torch.Tensor):
            z = item
            decomposition = decompose_latent(z.unsqueeze(0) if z.ndim == 3 else z)
            z_l = decomposition.z_l
            z_h = decomposition.z_h
        elif isinstance(item, dict):
            if "z" in item:
                z = item["z"]
                decomposition = decompose_latent(z.unsqueeze(0) if z.ndim == 3 else z)
                z_l = item.get("z_L", item.get("z_l", decomposition.z_l))
                z_h = item.get("z_H", item.get("z_h", decomposition.z_h))
            elif ("z_L" in item or "z_l" in item) and ("z_H" in item or "z_h" in item):
                z_l = item.get("z_L", item.get("z_l"))
                z_h = item.get("z_H", item.get("z_h"))
                z = None
            else:
                raise KeyError(f"{self.paths[index]} must contain z, or both z_L and z_H")
        else:
            raise TypeError(f"unsupported latent cache item type: {type(item)!r}")

        z_l = self._squeeze_batch(z_l, "z_L", index)
        z_h = self._squeeze_batch(z_h, "z_H", index)
        output = {"z_L": z_l.float(), "z_H": z_h.float()}
        if z is not None:
            output["z"] = self._squeeze_batch(z, "z", index).float()
        return output

    def _squeeze_batch(self, tensor: torch.Tensor, name: str, index: int) -> torch.Tensor:
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"expected {name} shape [C, H, W], got {tuple(tensor.shape)} from {self.paths[index]}")
        return tensor
