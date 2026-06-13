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
from src.micro.micro_rama_categorical import build_categorical_micro_rama_net
from src.modules.ema import EMA
from src.macro.factory import build_macro_flow_model
from src.modules.micro_rama import build_context_encoder, build_micro_rama_net, sample_micro_latent
from src.modules.rama import unpatchify
from src.dataset.vae import decode_latents, load_sd_vae
from src.dataset.latent_decomposition import reconstruct_from_decomposition, reconstruct_low_freq
from src.rama.projector import RAMAProjector
from src.rama.tokenizer import build_tokenizer_from_config, load_tokenizer_config


def resolve_vae_checkpoint(vae_cfg: dict[str, object], override: str | None) -> str:
    if override:
        return override
    local_checkpoint = vae_cfg.get("local_checkpoint")
    if local_checkpoint and Path(str(local_checkpoint)).exists():
        return str(local_checkpoint)
    return str(vae_cfg.get("checkpoint_id", "stabilityai/sd-vae-ft-mse"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample full images from macro flow plus micro RAMA.")
    parser.add_argument("--macro-checkpoint", required=True)
    parser.add_argument("--micro-checkpoint", required=True)
    parser.add_argument("--macro-config", default=None)
    parser.add_argument("--micro-config", default=None)
    parser.add_argument("--bases", default="cache/rama_bases_p256_d16.pt")
    parser.add_argument("--tokenizer-config", default="cache/rama_tokenizer_config.pt")
    parser.add_argument("--out", default="outputs/full_samples/generated_zL_macro_plus_micro.png")
    parser.add_argument("--macro-out", default="outputs/full_samples/generated_zL_macro_only.png")
    parser.add_argument("--micro-out", default="outputs/full_samples/generated_zL_micro_only.png")
    parser.add_argument("--comparison-out", default="outputs/full_samples/generated_zL_macro_micro_full.png")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sampler", choices=("heun", "euler", "shortcut"), default="heun")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use-argmax", action="store_true")
    parser.add_argument("--micro-type", choices=("auto", "categorical", "continuous"), default="auto")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Gaussian base-noise scale for continuous micro RAMA.")
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


def sample_tokens(logits: torch.Tensor, temperature: float, use_argmax: bool) -> torch.Tensor:
    if use_argmax:
        return logits.argmax(dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    return torch.multinomial(flat, num_samples=1).reshape(logits.shape[:-1])


def resolve_patch_size(config: dict[str, object]) -> int:
    micro_latent_cfg = config.get("micro_latent", {})
    rama_cfg = config.get("rama", {})
    micro_cfg = config.get("micro", config.get("micro_rama_net", {}))
    return int(
        micro_latent_cfg.get(
            "patch_size",
            rama_cfg.get("patch_size", micro_cfg.get("patch_size", 2)),
        )
    )


def resolve_micro_type(config: dict[str, object], override: str) -> str:
    if override != "auto":
        return override
    if "micro_type" in config:
        micro_type = str(config["micro_type"])
        if micro_type == "conditional_rq_nsf" or micro_type == "nflows":
            micro_type = "continuous"
        if micro_type not in {"categorical", "continuous"}:
            raise ValueError(f"unsupported micro type: {micro_type}")
        return micro_type
    micro_cfg = config.get("micro", {})
    micro_rama_cfg = config.get("micro_rama_net", {})
    micro_type = str(micro_cfg.get("type", micro_rama_cfg.get("type", "categorical")))
    if micro_type == "conditional_rq_nsf":
        micro_type = "continuous"
    if micro_type == "nflows":
        micro_type = "continuous"
    if bool(config.get("micro_continuous", {}).get("enabled", False)):
        micro_type = "continuous"
    if micro_type not in {"categorical", "continuous"}:
        raise ValueError(f"unsupported micro type: {micro_type}")
    return micro_type


def resolve_residual_shape(
    macro_model: torch.nn.Module,
    macro_config: dict[str, object],
    micro_config: dict[str, object],
    patch_size: int,
    projector: RAMAProjector,
) -> tuple[int, int, int]:
    micro_latent_cfg = micro_config.get("micro_latent", {})
    residual_shape = micro_latent_cfg.get("residual_shape")
    if residual_shape is not None:
        channels, height, width = (int(value) for value in residual_shape)
    else:
        full_latent_size = int(
            macro_config.get("evaluation", {}).get(
                "full_latent_size",
                int(macro_model.resolution) * int(macro_config.get("decomposition", {}).get("macro_downsample_factor", 2)),
            )
        )
        channels = int(getattr(macro_model, "in_channels"))
        height = full_latent_size
        width = full_latent_size

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"residual latent shape {height}x{width} must divide by patch_size={patch_size}")
    expected_patches = (height // patch_size) * (width // patch_size)
    if projector.num_patches != expected_patches:
        raise ValueError(f"RAMA bases have {projector.num_patches} patches, expected {expected_patches} for {height}x{width}")
    expected_patch_dim = channels * patch_size * patch_size
    if projector.patch_dim != expected_patch_dim:
        raise ValueError(f"RAMA bases patch_dim={projector.patch_dim}, expected {expected_patch_dim}")
    return channels, height, width


def main() -> None:
    args = parse_args()
    macro_checkpoint = torch.load(args.macro_checkpoint, map_location="cpu")
    micro_checkpoint = torch.load(args.micro_checkpoint, map_location="cpu")
    macro_config = load_config(args.macro_config, macro_checkpoint)
    micro_config = load_config(args.micro_config, micro_checkpoint)

    macro_model = build_macro_flow_model(macro_config.get("macro_flow_model", {})).to(args.device)
    macro_model.load_state_dict(macro_checkpoint["model"])
    if macro_checkpoint.get("ema") is not None:
        ema = EMA(macro_model, decay=float(macro_config.get("training", {}).get("ema_decay", 0.9999)))
        ema.load_state_dict(macro_checkpoint["ema"])
        ema.copy_to(macro_model)
    macro_model.eval()

    micro_type = resolve_micro_type(micro_config, args.micro_type)
    tokenizer = None
    if micro_type == "categorical":
        tokenizer = build_tokenizer_from_config(load_tokenizer_config(args.tokenizer_config))
    context_encoder = build_context_encoder(micro_config.get("context_encoder", {})).to(args.device)
    if micro_type == "categorical":
        micro_cfg = micro_config.get("micro", micro_config.get("micro_rama_net", {}))
        micro_model = build_categorical_micro_rama_net(micro_cfg, num_bins=tokenizer.num_bins).to(args.device)
    else:
        micro_cfg = micro_config.get("micro_continuous", micro_config.get("micro_rama_net", {}))
        micro_model = build_micro_rama_net(micro_cfg).to(args.device)
    context_encoder.load_state_dict(micro_checkpoint["context_encoder"])
    micro_model.load_state_dict(micro_checkpoint["micro_model"])
    context_encoder.eval()
    micro_model.eval()

    bases = torch.load(args.bases, map_location="cpu").float()
    projector = RAMAProjector(bases).to(args.device)
    projector.requires_grad_(False)
    patch_size = resolve_patch_size(micro_config)
    latent_channels, latent_height, latent_width = resolve_residual_shape(
        macro_model,
        macro_config,
        micro_config,
        patch_size,
        projector,
    )

    shape = (args.num_samples, macro_model.in_channels, macro_model.resolution, macro_model.resolution)
    z_l = sample_macro_latents(macro_model, shape=shape, method=args.sampler, num_steps=args.steps, device=args.device)
    if micro_type == "categorical":
        if tokenizer is None:
            raise RuntimeError("categorical micro sampling requires a tokenizer")
        context = context_encoder(z_l)
        logits = micro_model(context)
        tokens = sample_tokens(logits, temperature=args.temperature, use_argmax=args.use_argmax)
        y_hat = tokenizer.dequantize(tokens)
        patches_hat = projector.inverse(y_hat)
        z_h_hat = unpatchify(
            patches_hat,
            channels=latent_channels,
            height=latent_height,
            width=latent_width,
            patch_size=patch_size,
        )
    else:
        z_h_hat = sample_micro_latent(
            z_l,
            context_encoder,
            micro_model,
            projector.bases,
            latent_channels=latent_channels,
            latent_height=latent_height,
            latent_width=latent_width,
            patch_size=patch_size,
            noise_scale=args.noise_scale,
        )
    z_hat = reconstruct_from_decomposition(z_l, z_h_hat)

    vae_cfg = macro_config.get("vae", {})
    vae_checkpoint = resolve_vae_checkpoint(vae_cfg, args.vae_checkpoint)
    vae = load_sd_vae(checkpoint=str(vae_checkpoint), cache_dir=args.cache_dir, dtype=args.dtype, device=args.device)
    macro_images = decode_latents(vae, reconstruct_low_freq(z_l))
    micro_images = decode_latents(vae, z_h_hat)
    full_images = decode_latents(vae, z_hat)

    nrow = max(1, int(args.num_samples**0.5))
    macro_out = Path(args.macro_out)
    macro_out.parent.mkdir(parents=True, exist_ok=True)
    save_image((macro_images.float().clamp(-1, 1) + 1.0) / 2.0, macro_out, nrow=nrow)
    micro_out = Path(args.micro_out)
    micro_out.parent.mkdir(parents=True, exist_ok=True)
    save_image((micro_images.float().clamp(-1, 1) + 1.0) / 2.0, micro_out, nrow=nrow)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((full_images.float().clamp(-1, 1) + 1.0) / 2.0, out, nrow=nrow)
    comparison_out = Path(args.comparison_out)
    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    comparison_images = torch.cat([macro_images, micro_images, full_images], dim=0)
    save_image((comparison_images.float().clamp(-1, 1) + 1.0) / 2.0, comparison_out, nrow=args.num_samples)
    print(f"saved {macro_out}")
    print(f"saved {micro_out}")
    print(f"saved {out}")
    print(f"saved {comparison_out}")


if __name__ == "__main__":
    main()
