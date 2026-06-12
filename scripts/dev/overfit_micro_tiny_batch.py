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
from src.data.latent_dataset import CachedMicroLatentDataset
from src.modules.micro_rama import build_context_encoder
from src.modules.rama import patchify
from src.rama.projector import RAMAProjector
from src.rama.tokenizer import RAMATokenizer, build_tokenizer_from_config, load_tokenizer_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that categorical micro RAMA can overfit a tiny latent set.")
    parser.add_argument("--config", default="configs/debug_6gb_micro.yaml")
    parser.add_argument("--latents", default=None)
    parser.add_argument("--bases", default="cache/rama_bases_p256_d16.pt")
    parser.add_argument("--tokenizer-config", default="cache/rama_tokenizer_config.pt")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    latent_dir = args.latents or config.get("latents", {}).get("output_dir", "data/latents")
    dataset = CachedMicroLatentDataset(latent_dir)
    subset = Subset(dataset, list(range(min(args.num_images, len(dataset)))))
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    tokenizer_config = Path(args.tokenizer_config)
    tokenizer = (
        build_tokenizer_from_config(load_tokenizer_config(str(tokenizer_config)))
        if tokenizer_config.exists()
        else RAMATokenizer(num_bins=int(config.get("tokenizer", {}).get("num_bins", 256)))
    )
    bases = torch.load(args.bases, map_location="cpu").float()
    projector = RAMAProjector(bases).to(args.device)
    projector.requires_grad_(False)

    context_encoder = build_context_encoder(config.get("context_encoder", {})).to(args.device)
    micro_model = build_categorical_micro_rama_net(
        config.get("micro", config.get("micro_rama_net", {})),
        num_bins=tokenizer.num_bins,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(micro_model.parameters()),
        lr=args.lr,
    )

    patch_size = int(config.get("rama", config.get("micro_latent", {})).get("patch_size", 2))
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
        logits = micro_model(context_encoder(z_l))
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
