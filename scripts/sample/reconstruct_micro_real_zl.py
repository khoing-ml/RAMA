from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.micro.micro_rama_categorical import build_categorical_micro_rama_net
from src.dataset.latent_dataset import CachedMicroLatentDataset
from src.dataset.latent_decomposition import decompose_latent
from src.modules.micro_rama import build_context_encoder
from src.modules.rama import unpatchify
from src.dataset.vae import decode_latents, load_sd_vae
from src.rama.projector import RAMAProjector
from src.rama.tokenizer import build_tokenizer_from_config, load_tokenizer_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct residuals from real z_L with categorical micro RAMA.")
    parser.add_argument("--micro-checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--latent-cache", default="data/latents")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--bases", default="cache/rama_bases_p256_d16.pt")
    parser.add_argument("--tokenizer-config", default="cache/rama_tokenizer_config.pt")
    parser.add_argument("--out", default="outputs/micro_reconstructions/real_zL_macro_vs_micro_argmax.png")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use-sampling", action="store_true")
    parser.add_argument("--vae-checkpoint", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path: str | None, checkpoint: dict[str, object]) -> dict[str, object]:
    if path is None:
        return checkpoint.get("config", {})
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_tokens(logits: torch.Tensor, use_sampling: bool, temperature: float) -> torch.Tensor:
    if not use_sampling:
        return logits.argmax(dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(logits.shape[:-1])


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


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.micro_checkpoint, map_location="cpu")
    config = load_config(args.config, checkpoint)
    tokenizer = build_tokenizer_from_config(load_tokenizer_config(args.tokenizer_config))

    context_encoder = build_context_encoder(config.get("context_encoder", {})).to(args.device)
    micro_model = build_categorical_micro_rama_net(
        config.get("micro", config.get("micro_rama_net", {})),
        num_bins=tokenizer.num_bins,
    ).to(args.device)
    context_encoder.load_state_dict(checkpoint["context_encoder"])
    micro_model.load_state_dict(checkpoint["micro_model"])
    context_encoder.eval()
    micro_model.eval()

    bases = torch.load(args.bases, map_location="cpu").float()
    projector = RAMAProjector(bases).to(args.device)
    patch_size = resolve_patch_size(config)
    dataset = CachedMicroLatentDataset(args.latent_cache)
    item = dataset[args.index]
    if "z" in item:
        z = item["z"].unsqueeze(0).to(args.device)
        decomposition = decompose_latent(z)
        z_l = decomposition.z_l
        z_l_up = decomposition.z_l_up
    else:
        z_l = item["z_L"].unsqueeze(0).to(args.device)
        z_h = item["z_H"].unsqueeze(0).to(args.device)
        z_l_up = F.interpolate(z_l, size=z_h.shape[-2:], mode="bilinear", align_corners=False)
        z = z_l_up + z_h

    with torch.no_grad():
        logits = micro_model(context_encoder(z_l))
        tokens = choose_tokens(logits, args.use_sampling, args.temperature)
        y_hat = tokenizer.dequantize(tokens)
        patches_hat = projector.inverse(y_hat)
        z_h_hat = unpatchify(
            patches_hat,
            channels=z_l_up.shape[1],
            height=z_l_up.shape[2],
            width=z_l_up.shape[3],
            patch_size=patch_size,
        )
        z_hat = z_l_up + z_h_hat
        vae = load_sd_vae(checkpoint=args.vae_checkpoint, cache_dir=args.cache_dir, dtype=args.dtype, device=args.device)
        images = torch.cat([decode_latents(vae, z), decode_latents(vae, z_l_up), decode_latents(vae, z_hat)], dim=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((images.float().clamp(-1, 1) + 1.0) / 2.0, out, nrow=3)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
