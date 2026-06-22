from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.micro.loss import categorical_micro_loss, categorical_micro_metrics
from src.micro.micro_rama_categorical import build_categorical_micro_rama_net
from src.dataset.latent_dataset import CachedMicroLatentDataset
from src.modules.micro_rama import build_context_encoder
from src.modules.rama import patchify
from src.rama.projector import RAMAProjector
from src.rama.tokenizer import RAMATokenizer, build_tokenizer_from_config, load_tokenizer_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that categorical micro RAMA can overfit a tiny latent set.")
    parser.add_argument("--config", default="configs/debug_6gb_micro.yaml")
    parser.add_argument("--latents", default=None)
    parser.add_argument("--bases", default=None, help="path to RAMA bases .pt file; auto-detected from cache/ if omitted")
    parser.add_argument("--tokenizer-config", default="cache/rama_tokenizer_config.pt")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--context-mode",
        choices=["trained", "oracle", "zero"],
        default="trained",
        help=(
            "trained: normal encoder+micro joint training (default). "
            "oracle: bypass encoder — context is a fixed linear projection of z_H patches "
            "(upper bound; if micro can't overfit here, micro is the bottleneck). "
            "zero: feed all-zeros context — no encoder signal at all (lower bound baseline)."
        ),
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class _SyntheticMicroDataset(torch.utils.data.Dataset):
    """Random z_L / z_H tensors for smoke-testing without real data."""

    def __init__(self, num_images: int, residual_shape: list[int]) -> None:
        C, H, W = int(residual_shape[0]), int(residual_shape[1]), int(residual_shape[2])
        self.z_h = torch.randn(num_images, C, H, W)
        self.z_l = torch.randn(num_images, C, H // 2, W // 2)

    def __len__(self) -> int:
        return len(self.z_h)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"z_L": self.z_l[index], "z_H": self.z_h[index]}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    latent_dir = args.latents or config.get("latents", {}).get("output_dir", "data/latents")

    residual_shape = config.get("micro_latent", {}).get("residual_shape", [4, 32, 32])
    try:
        dataset = CachedMicroLatentDataset(latent_dir)
        print(f"Loaded {len(dataset)} latents from {latent_dir}")
    except FileNotFoundError:
        print(f"[warn] no latents found at {latent_dir} — using synthetic random tensors")
        print(f"       (results are only meaningful for diagnosing model capacity, not data fit)")
        dataset = _SyntheticMicroDataset(args.num_images, residual_shape)

    subset = Subset(dataset, list(range(min(args.num_images, len(dataset)))))
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    tokenizer_config = Path(args.tokenizer_config)
    tokenizer = (
        build_tokenizer_from_config(load_tokenizer_config(str(tokenizer_config)))
        if tokenizer_config.exists()
        else RAMATokenizer(num_bins=int(config.get("tokenizer", {}).get("num_bins", 256)))
    )
    context_dim = int(config.get("context_encoder", {}).get("context_dim", 256))
    ml_cfg = config.get("micro_latent", {})
    patch_size = int(ml_cfg.get("patch_size", 2))
    residual_shape = ml_cfg.get("residual_shape", [4, 32, 32])
    C, H_res, W_res = int(residual_shape[0]), int(residual_shape[1]), int(residual_shape[2])
    patch_dim = C * patch_size * patch_size
    num_patches = (H_res // patch_size) * (W_res // patch_size)

    bases_path = Path(args.bases) if args.bases else None
    if bases_path is None:
        candidates = sorted(Path("cache").glob("rama_bases_*.pt")) if Path("cache").exists() else []
        bases_path = candidates[0] if candidates else None
    if bases_path and bases_path.exists():
        print(f"Loading bases from {bases_path}")
        bases = torch.load(bases_path, map_location="cpu").float()
    else:
        from src.modules.rama import make_orthogonal_bases
        print(f"[warn] no bases file found — generating random orthogonal bases (patch_dim={patch_dim}, num_patches={num_patches})")
        bases = make_orthogonal_bases(num_patches, patch_dim)
    projector = RAMAProjector(bases).to(args.device)
    projector.requires_grad_(False)

    micro_cfg = dict(config.get("micro", config.get("micro_rama_net", {})))
    micro_cfg["patch_dim"] = patch_dim
    micro_model = build_categorical_micro_rama_net(micro_cfg, num_bins=tokenizer.num_bins).to(args.device)

    if args.context_mode == "trained":
        enc_cfg = dict(config.get("context_encoder", {}))
        enc_cfg["grid_size"] = [H_res // patch_size, W_res // patch_size]
        enc_cfg["patch_size"] = patch_size
        context_encoder = build_context_encoder(enc_cfg).to(args.device)
        opt_params = list(context_encoder.parameters()) + list(micro_model.parameters())
        print(f"[mode=trained] training encoder + micro jointly")
    elif args.context_mode == "oracle":
        # Fixed linear projection of the actual patches → context.
        # Encoder is bypassed; micro sees a perfect (but simple) context derived from z_H.
        # If micro can't overfit here, it's underpowered or the task is too hard regardless.
        oracle_proj = torch.nn.Linear(patch_dim, context_dim, bias=False).to(args.device)
        oracle_proj.requires_grad_(False)  # fixed — not trained
        context_encoder = None
        opt_params = list(micro_model.parameters())
        print(f"[mode=oracle] context = fixed linear projection of z_H patches (encoder bypassed)")
        print(f"  → if micro fails to overfit here, micro model is the bottleneck")
        print(f"  → if micro overfits here but not in [trained] mode, encoder is the bottleneck")
    else:  # zero
        context_encoder = None
        opt_params = list(micro_model.parameters())
        print(f"[mode=zero] context = all zeros (lower bound — no encoder signal)")

    optimizer = torch.optim.AdamW(opt_params, lr=args.lr)

    iterator = iter(dataloader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        z_l = batch["z_L"].to(args.device).detach()
        z_h = batch["z_H"].to(args.device).detach()
        patches = patchify(z_h, patch_size=patch_size)
        tokens = tokenizer.quantize(projector.project(patches))

        if args.context_mode == "trained":
            ctx = context_encoder(z_l)
        elif args.context_mode == "oracle":
            with torch.no_grad():
                # patches: [B, P, patch_dim] → oracle_proj → [B, P, context_dim]
                ctx = oracle_proj(patches.float())
        else:  # zero
            B, P = patches.shape[:2]
            ctx = torch.zeros(B, P, context_dim, device=args.device)

        logits = micro_model(ctx)
        loss = categorical_micro_loss(logits, tokens, tokenizer.num_bins)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            metrics = categorical_micro_metrics(logits, tokens, tokenizer.num_bins)
            print(
                f"step={step} loss={loss.item():.6f} "
                f"token_acc={metrics['token_acc'].item():.4f} "
                f"top5={metrics['token_top5_acc'].item():.4f} "
                f"top10={metrics['token_top10_acc'].item():.4f} "
                f"within1={metrics['token_within_1'].item():.4f} "
                f"within2={metrics['token_within_2'].item():.4f}"
            )


if __name__ == "__main__":
    main()
