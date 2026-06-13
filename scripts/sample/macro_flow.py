from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.macro.sampler import sample_macro_latents
from src.modules.ema import EMA
from src.macro.factory import build_macro_flow_model
from src.dataset.vae import decode_latents, load_sd_vae
from src.dataset.latent_decomposition import reconstruct_low_freq


def resolve_vae_checkpoint(vae_cfg: dict[str, object], override: str | None) -> str:
    if override:
        return override
    local_checkpoint = vae_cfg.get("local_checkpoint")
    if local_checkpoint and Path(str(local_checkpoint)).exists():
        return str(local_checkpoint)
    return str(vae_cfg.get("checkpoint_id", "stabilityai/sd-vae-ft-mse"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample macro-only images from a trained macro flow checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Optional config override; checkpoint config is used by default.")
    parser.add_argument("--out", default="outputs/macro_samples/macro_only_samples.png")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sampler", choices=("heun", "euler", "shortcut"), default="heun")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--vae-checkpoint", default=None)
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path: str | None, checkpoint: dict[str, object]) -> dict[str, object]:
    if path is None:
        return checkpoint.get("config", {})
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_full_latent_size(config: dict[str, object], macro_resolution: int) -> int:
    evaluation_cfg = config.get("evaluation", {})
    if "full_latent_size" in evaluation_cfg:
        return int(evaluation_cfg["full_latent_size"])
    factor = int(config.get("decomposition", {}).get("macro_downsample_factor", 2))
    return int(macro_resolution) * factor


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(args.config, checkpoint)
    model = build_macro_flow_model(config.get("macro_flow_model", {})).to(args.device)
    model.load_state_dict(checkpoint["model"])
    if args.use_ema and checkpoint.get("ema") is not None:
        ema = EMA(model, decay=float(config.get("training", {}).get("ema_decay", 0.9999)))
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    model.eval()

    shape = (args.num_samples, model.in_channels, model.resolution, model.resolution)
    z_l = sample_macro_latents(model, shape=shape, method=args.sampler, num_steps=args.steps, device=args.device)
    full_latent_size = resolve_full_latent_size(config, model.resolution)
    vae_cfg = config.get("vae", {})
    vae_checkpoint = resolve_vae_checkpoint(vae_cfg, args.vae_checkpoint)
    vae = load_sd_vae(checkpoint=str(vae_checkpoint), cache_dir=args.cache_dir, dtype=args.dtype, device=args.device)
    images = decode_latents(vae, reconstruct_low_freq(z_l))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((images.float().clamp(-1, 1) + 1.0) / 2.0, out, nrow=max(1, int(args.num_samples**0.5)))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
