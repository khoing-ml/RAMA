from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
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
                # Always re-decompose from z so z_H uses the current decomposition
                # (ignores any stale z_L/z_H that may be stored alongside z in cache).
                z = item["z"]
                decomposition = decompose_latent(z.unsqueeze(0) if z.ndim == 3 else z)
                z_l = decomposition.z_l
                z_h = decomposition.z_h
            elif ("z_L" in item or "z_l" in item) and ("z_H" in item or "z_h" in item):
                z_l = item.get("z_L", item.get("z_l"))
                z_h = item.get("z_H", item.get("z_h"))
                z_l_3d = z_l.squeeze(0) if z_l.ndim == 4 else z_l
                z_h_3d = z_h.squeeze(0) if z_h.ndim == 4 else z_h
                # Detect stale Haar cache: z_H has 3× the channels of z_L.
                # Reconstruct z via Haar IDWT, then re-decompose with Laplacian.
                if z_h_3d.shape[0] != z_l_3d.shape[0]:
                    C = z_l_3d.shape[0]
                    lh, hl, hh = z_h_3d[:C], z_h_3d[C:2*C], z_h_3d[2*C:]
                    zl = z_l_3d.unsqueeze(0)
                    H, W = zl.shape[-2], zl.shape[-1]
                    z_full = torch.empty(1, C, H * 2, W * 2, device=zl.device, dtype=zl.dtype)
                    z_full[:, :, 0::2, 0::2] = zl + lh + hl + hh
                    z_full[:, :, 0::2, 1::2] = zl - lh + hl - hh
                    z_full[:, :, 1::2, 0::2] = zl + lh - hl - hh
                    z_full[:, :, 1::2, 1::2] = zl - lh - hl + hh
                    decomposition = decompose_latent(z_full)
                    z_l = decomposition.z_l
                    z_h = decomposition.z_h
                    z = z_full
                else:
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
