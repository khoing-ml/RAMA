from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset.latent_decomposition import reconstruct_from_decomposition
from src.dataset.latent_dataset import CachedMicroLatentDataset
from src.dataset.vae import decode_latents
from src.evaluation.fid import FIDStats, InceptionFID, stats_from_feature_batches
from src.macro.factory import build_macro_flow_model
from src.micro.micro_rama_categorical import build_categorical_micro_rama_net
from src.modules.ema import EMA
from src.modules.micro_rama import build_context_encoder, build_micro_rama_net
from src.modules.rama import make_orthogonal_bases
from src.rama.tokenizer import RAMATokenizer


def save_fid_stats(stats: FIDStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": stats.mean, "covariance": stats.covariance, "num_samples": stats.num_samples}, path)


def load_fid_stats(path: str | Path) -> FIDStats:
    data = torch.load(path, map_location="cpu")
    return FIDStats(mean=data["mean"], covariance=data["covariance"], num_samples=data["num_samples"])


def resolve_vae_checkpoint(vae_cfg: dict, override: str | None = None) -> str:
    if override:
        return override
    local_checkpoint = vae_cfg.get("local_checkpoint")
    if local_checkpoint and Path(str(local_checkpoint)).exists():
        return str(local_checkpoint)
    return str(vae_cfg.get("checkpoint_id", "stabilityai/sd-vae-ft-mse"))


def resolve_micro_type(config: dict, override: str) -> str:
    if override != "auto":
        micro_type = override
    elif "micro_type" in config:
        micro_type = str(config["micro_type"])
    else:
        micro_type = str(config.get("micro", config.get("micro_rama_net", {})).get("type", "categorical"))
    if micro_type in {"conditional_rq_nsf", "nflows"}:
        micro_type = "continuous"
    if micro_type not in {"categorical", "continuous"}:
        raise ValueError(f"unsupported micro type: {micro_type}")
    return micro_type


def resolve_patch_size(config: dict) -> int:
    micro_latent_cfg = config.get("micro_latent", {})
    rama_cfg = config.get("rama", {})
    micro_cfg = config.get("micro", config.get("micro_rama_net", {}))
    return int(micro_latent_cfg.get("patch_size", rama_cfg.get("patch_size", micro_cfg.get("patch_size", 2))))


def sample_tokens(logits: torch.Tensor, temperature: float, use_argmax: bool) -> torch.Tensor:
    if use_argmax:
        return logits.argmax(dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    return torch.multinomial(flat, num_samples=1).reshape(logits.shape[:-1])


def load_or_make_bases(config: dict, override_path: str | None = None) -> torch.Tensor:
    basis_path = Path(str(override_path or config.get("cache_path", config.get("bases_path", "cache/rama_bases_p256_d16.pt"))))
    if basis_path.exists():
        return torch.load(basis_path, map_location="cpu").float()
    bases = make_orthogonal_bases(
        num_patches=int(config.get("num_patches", 256)),
        patch_dim=int(config.get("patch_dim", 16)),
        seed=int(config.get("seed", 1234)),
        device="cpu",
    )
    basis_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bases, basis_path)
    return bases.float()


def load_macro_model(
    checkpoint_path: str,
    device: str | torch.device,
    use_ema: bool,
) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = build_macro_flow_model(config.get("macro_flow_model", {}))
    model.load_state_dict(checkpoint["model"])
    if use_ema and checkpoint.get("ema") is not None:
        ema = EMA(model, decay=float(config.get("training", {}).get("ema_decay", 0.9999)))
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    model.to(device).eval()
    return model, config


def load_micro_models(
    checkpoint_path: str,
    micro_type: str,
    tokenizer: RAMATokenizer | None,
    device: str | torch.device,
) -> tuple[nn.Module, nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    context_encoder = build_context_encoder(config.get("context_encoder", {}))
    if micro_type == "categorical":
        micro_model = build_categorical_micro_rama_net(
            config.get("micro", config.get("micro_rama_net", {})),
            num_bins=tokenizer.num_bins if tokenizer is not None else None,
        )
    else:
        micro_model = build_micro_rama_net(config.get("micro_continuous", config.get("micro_rama_net", {})))
    context_encoder.load_state_dict(checkpoint["context_encoder"])
    micro_model.load_state_dict(checkpoint["micro_model"])
    context_encoder.to(device).eval()
    micro_model.to(device).eval()
    return context_encoder, micro_model, config


@torch.no_grad()
def collect_real_fid_stats(
    dataset: CachedMicroLatentDataset,
    vae: nn.Module,
    fid_model: InceptionFID,
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> FIDStats:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    feature_batches: list[torch.Tensor] = []
    remaining = num_samples
    for batch in loader:
        if remaining <= 0:
            break
        z_l = batch["z_L"][:remaining].to(device)
        z_h = batch["z_H"][:remaining].to(device)
        z_real = reconstruct_from_decomposition(z_l, z_h)
        feature_batches.append(fid_model(decode_latents(vae, z_real)))
        remaining -= z_l.shape[0]
    return stats_from_feature_batches(feature_batches)
