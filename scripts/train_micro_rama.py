from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.latent_dataset import CachedMicroLatentDataset
from src.modules.micro_rama import build_context_encoder, build_micro_rama_net, micro_nll_loss
from src.modules.rama import make_orthogonal_bases, patchify, rama_project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the micro latent RAMA model.")
    parser.add_argument("--config", default="configs/debug_6gb_micro.yaml")
    parser.add_argument("--latents", default=None, help="Override latents.output_dir from config.")
    parser.add_argument("--out", default="outputs/micro_rama")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-process batch size.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.total_steps for debug runs.")
    parser.add_argument("--disable-wandb", action="store_true", help="Run without initializing Weights & Biases.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint_path(out_dir: Path, step: int) -> Path:
    return out_dir / "checkpoints" / f"step_{step:08d}.pt"


def save_checkpoint(
    path: Path,
    step: int,
    context_encoder: torch.nn.Module,
    micro_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "context_encoder": context_encoder.state_dict(),
            "micro_model": micro_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def load_or_make_bases(config: dict[str, object]) -> torch.Tensor:
    basis_path = Path(str(config.get("cache_path", "cache/rama_bases_p256_d16.pt")))
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config.get("training", {})
    logging_cfg = config.get("logging", {})
    micro_latent_cfg = config.get("micro_latent", {})

    mixed_precision = str(training.get("precision", "fp32"))
    if mixed_precision not in {"fp16", "bf16"}:
        mixed_precision = "no"

    tracker = None if args.disable_wandb else logging_cfg.get("tracker")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=mixed_precision,
        log_with="wandb" if tracker == "wandb" else None,
    )

    out_dir = Path(args.out)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    latent_dir = args.latents or config.get("latents", {}).get("output_dir", "data/latents")
    dataset = CachedMicroLatentDataset(latent_dir)
    batch_size = args.batch_size or int(training.get("batch_size_per_gpu", 64))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if len(dataloader) == 0:
        raise ValueError("latent dataset is smaller than the per-process batch size; lower batch size or add latents")

    context_encoder = build_context_encoder(config.get("context_encoder", {}))
    micro_model = build_micro_rama_net(config.get("micro_rama_net", {}))
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(micro_model.parameters()),
        lr=float(training.get("learning_rate", training.get("lr", 2.0e-4))),
        betas=tuple(training.get("betas", [0.9, 0.999])),
        weight_decay=float(training.get("weight_decay", 1.0e-4)),
    )
    bases = load_or_make_bases(config.get("rama_bases", {}))

    context_encoder, micro_model, optimizer, dataloader = accelerator.prepare(
        context_encoder,
        micro_model,
        optimizer,
        dataloader,
    )
    bases = bases.to(accelerator.device)
    bases.requires_grad_(False)

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        accelerator.unwrap_model(context_encoder).load_state_dict(checkpoint["context_encoder"])
        accelerator.unwrap_model(micro_model).load_state_dict(checkpoint["micro_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    if accelerator.is_main_process and tracker == "wandb":
        accelerator.init_trackers(project_name=str(logging_cfg.get("project", "rama")), config=config)

    patch_size = int(micro_latent_cfg.get("patch_size", 2))
    context_noise_sigma = float(config.get("context_encoder", {}).get("context_noise_sigma", 0.03))
    grad_clip = float(training.get("grad_clip", 1.0))
    total_steps = args.max_steps or int(training.get("total_steps", 200000))
    log_every = int(logging_cfg.get("log_every_steps", 100))
    checkpoint_every = int(logging_cfg.get("checkpoint_every_steps", 10000))

    step = start_step
    context_encoder.train()
    micro_model.train()

    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break
            z_l = batch["z_L"]
            z_h = batch["z_H"]

            with accelerator.accumulate(micro_model):
                z_l_input = z_l
                if context_noise_sigma > 0:
                    z_l_input = z_l_input + context_noise_sigma * torch.randn_like(z_l_input)

                patches = patchify(z_h, patch_size=patch_size)
                y = rama_project(patches, bases)
                context = context_encoder(z_l_input)
                eps, logabsdet = micro_model(y, context)
                loss = micro_nll_loss(eps, logabsdet)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        list(context_encoder.parameters()) + list(micro_model.parameters()),
                        grad_clip,
                    )
                else:
                    grad_norm = torch.tensor(0.0, device=accelerator.device)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % log_every == 0:
                logs = {
                    "train/loss": loss.detach().float().item(),
                    "train/grad_norm": float(grad_norm),
                    "train/y_abs_mean": y.detach().abs().float().mean().item(),
                    "train/logabsdet_mean": logabsdet.detach().float().mean().item(),
                }
                accelerator.log(logs, step=step)
                accelerator.print(
                    f"step={step} loss={logs['train/loss']:.6f} "
                    f"grad_norm={logs['train/grad_norm']:.4f} "
                    f"y_abs_mean={logs['train/y_abs_mean']:.4f} "
                    f"logabsdet_mean={logs['train/logabsdet_mean']:.4f}"
                )

            if accelerator.is_main_process and step % checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_path(out_dir, step),
                    step,
                    accelerator.unwrap_model(context_encoder),
                    accelerator.unwrap_model(micro_model),
                    optimizer,
                    copy.deepcopy(config),
                )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            checkpoint_path(out_dir, step),
            step,
            accelerator.unwrap_model(context_encoder),
            accelerator.unwrap_model(micro_model),
            optimizer,
            copy.deepcopy(config),
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
