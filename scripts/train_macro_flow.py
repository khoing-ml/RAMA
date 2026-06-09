from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.ema import EMA
from src.evaluation.fid import InceptionFID, calculate_fid, stats_from_feature_batches
from src.flow.sampler import sample_macro_latents
from src.modules.flow_matching import flow_matching_loss
from src.modules.latent_dataset import CachedLatentDataset
from src.modules.unet_flow import build_unet_flow
from src.modules.vae_utils import decode_latents, load_sd_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the macro latent rectified-flow model.")
    parser.add_argument("--config", default="configs/celeba256_sdvae_macro.yaml")
    parser.add_argument("--latents", default=None, help="Override latents.output_dir from config.")
    parser.add_argument("--out", default="outputs/macro_flow")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-process batch size.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.total_steps for debug runs.")
    parser.add_argument("--fid-every", type=int, default=None, help="Override evaluation.fid_every_steps; <=0 disables FID.")
    parser.add_argument("--fid-num-samples", type=int, default=None, help="Override evaluation.fid_num_samples.")
    parser.add_argument("--disable-wandb", action="store_true", help="Run without initializing Weights & Biases.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint_path(out_dir: Path, step: int) -> Path:
    return out_dir / "checkpoints" / f"step_{step:08d}.pt"


def resolve_vae_checkpoint(vae_cfg: dict[str, object]) -> str:
    local_checkpoint = vae_cfg.get("local_checkpoint")
    if local_checkpoint and Path(str(local_checkpoint)).exists():
        return str(local_checkpoint)
    return str(vae_cfg.get("checkpoint_id", "stabilityai/sd-vae-ft-mse"))


def load_real_latent_for_fid(path: Path, full_latent_size: int) -> torch.Tensor:
    item = torch.load(path, map_location="cpu")
    if isinstance(item, torch.Tensor):
        z = item
    elif isinstance(item, dict) and "z" in item:
        z = item["z"]
    elif isinstance(item, dict) and ("z_L" in item or "z_l" in item):
        z_l = item.get("z_L", item.get("z_l"))
        if z_l.ndim == 3:
            z_l = z_l.unsqueeze(0)
        return F.interpolate(z_l.float(), size=(full_latent_size, full_latent_size), mode="bilinear", align_corners=False)
    else:
        raise KeyError(f"{path} must contain z, z_L, or z_l for FID evaluation")

    if z.ndim == 3:
        z = z.unsqueeze(0)
    return z.float()


def save_checkpoint(
    path: Path,
    step: int,
    model: torch.nn.Module,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


@torch.no_grad()
def evaluate_macro_fid(
    model: torch.nn.Module,
    dataset: CachedLatentDataset,
    vae: torch.nn.Module,
    fid_model: InceptionFID,
    num_samples: int,
    batch_size: int,
    sampler: str,
    sampler_steps: int,
    full_latent_size: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    real_batches: list[torch.Tensor] = []
    fake_batches: list[torch.Tensor] = []

    real_latents: list[torch.Tensor] = []
    for path in dataset.paths[:num_samples]:
        z = load_real_latent_for_fid(path, full_latent_size)
        real_latents.append(z)
        if len(real_latents) == batch_size:
            real_images = decode_latents(vae, torch.cat(real_latents, dim=0).to(device))
            real_batches.append(fid_model(real_images))
            real_latents.clear()
    if real_latents:
        real_images = decode_latents(vae, torch.cat(real_latents, dim=0).to(device))
        real_batches.append(fid_model(real_images))

    remaining = sum(batch.shape[0] for batch in real_batches)
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        shape = (current_batch, model.in_channels, model.resolution, model.resolution)
        z_l = sample_macro_latents(model, shape=shape, method=sampler, num_steps=sampler_steps, device=str(device))
        z_l_up = F.interpolate(z_l, size=(full_latent_size, full_latent_size), mode="bilinear", align_corners=False)
        fake_images = decode_latents(vae, z_l_up)
        fake_batches.append(fid_model(fake_images))
        remaining -= current_batch

    if was_training:
        model.train()
    return calculate_fid(stats_from_feature_batches(real_batches), stats_from_feature_batches(fake_batches))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config.get("training", {})
    logging_cfg = config.get("logging", {})
    evaluation_cfg = config.get("evaluation", {})
    model_cfg = config.get("macro_flow_model", {})

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
    dataset = CachedLatentDataset(latent_dir)
    batch_size = args.batch_size or int(training.get("batch_size_per_gpu", training.get("per_gpu_batch_size", 64)))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if len(dataloader) == 0:
        raise ValueError(
            "latent dataset is smaller than the per-process batch size; "
            "lower training.batch_size_per_gpu or add more cached latents"
        )

    model = build_unet_flow(model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", training.get("lr", 2.0e-4))),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    ema = EMA(accelerator.unwrap_model(model), float(training.get("ema_decay", 0.9999))) if training.get("ema", True) else None
    start_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])
        start_step = int(checkpoint["step"])

    if accelerator.is_main_process and tracker == "wandb":
        accelerator.init_trackers(
            project_name=str(logging_cfg.get("project", "rama")),
            config=config,
            init_kwargs={
                "wandb": {
                    "entity": logging_cfg.get("entity"),
                    "name": logging_cfg.get("run_name"),
                }
            },
        )
        if logging_cfg.get("watch_model", False):
            import wandb

            wandb.watch(accelerator.unwrap_model(model), log="gradients", log_freq=int(logging_cfg.get("log_every_steps", 100)))

    total_steps = args.max_steps or int(training.get("total_steps", training.get("max_steps", 200000)))
    log_every = int(logging_cfg.get("log_every_steps", 100))
    checkpoint_every = int(logging_cfg.get("checkpoint_every_steps", 10000))
    fid_every = args.fid_every if args.fid_every is not None else int(evaluation_cfg.get("fid_every_steps", 0))
    fid_num_samples = args.fid_num_samples or int(evaluation_cfg.get("fid_num_samples", 512))
    fid_batch_size = int(evaluation_cfg.get("fid_batch_size", min(batch_size, 32)))
    fid_sampler = str(evaluation_cfg.get("sampler", "heun"))
    fid_sampler_steps = int(evaluation_cfg.get("sampler_steps", 50))
    full_latent_size = int(evaluation_cfg.get("full_latent_size", 32))
    vae = None
    fid_model = None
    if accelerator.is_main_process and fid_every > 0:
        vae_cfg = config.get("vae", {})
        vae = load_sd_vae(
            checkpoint=resolve_vae_checkpoint(vae_cfg),
            cache_dir=str(vae_cfg.get("cache_dir", ".cache/huggingface")),
            dtype=str(vae_cfg.get("dtype", "fp16")),
            device=str(accelerator.device),
        )
        fid_model = InceptionFID(accelerator.device)
    grad_clip = float(training.get("grad_clip", 1.0))
    step = start_step
    model.train()

    while step < total_steps:
        for z_l in dataloader:
            if step >= total_steps:
                break
            with accelerator.accumulate(model):
                loss, metrics = flow_matching_loss(model, z_l)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                else:
                    grad_norm = torch.tensor(0.0, device=accelerator.device)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None and accelerator.sync_gradients:
                    ema.update(accelerator.unwrap_model(model))

            step += 1
            if step % log_every == 0:
                logs = {
                    "train/loss": metrics["loss"].item(),
                    "train/target_v_norm": metrics["target_v_norm"].item(),
                    "train/pred_v_norm": metrics["pred_v_norm"].item(),
                    "train/grad_norm": float(grad_norm),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                accelerator.log(logs, step=step)
                accelerator.print(f"step={step} loss={logs['train/loss']:.6f} grad_norm={logs['train/grad_norm']:.4f}")

            if accelerator.is_main_process and fid_every > 0 and step % fid_every == 0:
                eval_model = accelerator.unwrap_model(model)
                backup = None
                if ema is not None:
                    backup = {
                        name: parameter.detach().clone()
                        for name, parameter in eval_model.named_parameters()
                        if parameter.requires_grad
                    }
                    ema.copy_to(eval_model)
                fid = evaluate_macro_fid(
                    eval_model,
                    dataset,
                    vae,
                    fid_model,
                    num_samples=min(fid_num_samples, len(dataset)),
                    batch_size=fid_batch_size,
                    sampler=fid_sampler,
                    sampler_steps=fid_sampler_steps,
                    full_latent_size=full_latent_size,
                    device=accelerator.device,
                )
                if backup is not None:
                    for name, parameter in eval_model.named_parameters():
                        if name in backup:
                            parameter.copy_(backup[name])
                accelerator.log({"eval/fid_macro": fid}, step=step)
                accelerator.print(f"step={step} fid_macro={fid:.4f}")
                model.train()

            if accelerator.is_main_process and step % checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_path(out_dir, step),
                    step,
                    accelerator.unwrap_model(model),
                    ema,
                    optimizer,
                    copy.deepcopy(config),
                )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            checkpoint_path(out_dir, step),
            step,
            accelerator.unwrap_model(model),
            ema,
            optimizer,
            copy.deepcopy(config),
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
