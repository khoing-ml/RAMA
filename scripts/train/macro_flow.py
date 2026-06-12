from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.ema import EMA
from src.evaluation.fid import InceptionFID, calculate_fid, stats_from_feature_batches
from src.macro.losses import flow_matching_loss, shortcut_matching_loss
from src.macro.sampler import sample_macro_latents
from src.data.latent_dataset import CachedLatentDataset, CachedMicroLatentDataset
from src.data.latent_decomposition import reconstruct_from_decomposition
from src.macro.factory import build_macro_flow_model
from src.data.vae import decode_latents, load_sd_vae


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
    fid_dataset: CachedMicroLatentDataset,
    vae: torch.nn.Module,
    fid_model: InceptionFID,
    num_samples: int,
    batch_size: int,
    sampler: str,
    sampler_steps: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    real_batches: list[torch.Tensor] = []
    fake_batches: list[torch.Tensor] = []

    loader = DataLoader(fid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    remaining = num_samples
    for batch in loader:
        if remaining <= 0:
            break
        z_l = batch["z_L"][:remaining].to(device)
        z_h = batch["z_H"][:remaining].to(device)
        z_real = reconstruct_from_decomposition(z_l, z_h)
        real_batches.append(fid_model(decode_latents(vae, z_real)))

        shape = (z_l.shape[0], model.in_channels, model.resolution, model.resolution)
        z_l_fake = sample_macro_latents(model, shape=shape, method=sampler, num_steps=sampler_steps, device=str(device))
        z_fake = reconstruct_from_decomposition(z_l_fake, z_h)
        fake_batches.append(fid_model(decode_latents(vae, z_fake)))
        remaining -= z_l.shape[0]

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
    objective = str(model_cfg.get("objective", training.get("objective", "rectified_flow")))

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

    model = build_macro_flow_model(model_cfg)
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

    vae = None
    fid_model = None
    fid_dataset = None
    if accelerator.is_main_process and fid_every > 0:
        vae_cfg = config.get("vae", {})
        vae = load_sd_vae(
            checkpoint=resolve_vae_checkpoint(vae_cfg),
            cache_dir=str(vae_cfg.get("cache_dir", ".cache/huggingface")),
            dtype=str(vae_cfg.get("dtype", "fp16")),
            device=str(accelerator.device),
        )
        fid_model = InceptionFID(accelerator.device)
        fid_dataset = CachedMicroLatentDataset(latent_dir)
    grad_clip = float(training.get("grad_clip", 1.0))
    step = start_step
    model.train()

    while step < total_steps:
        for z_l in dataloader:
            if step >= total_steps:
                break
            with accelerator.accumulate(model):
                if objective == "shortcut":
                    shortcut_cfg = config.get("shortcut", {})
                    loss, metrics = shortcut_matching_loss(
                        model,
                        z_l,
                        denoise_timesteps=int(shortcut_cfg.get("denoise_timesteps", 128)),
                        bootstrap_every=int(shortcut_cfg.get("bootstrap_every", 8)),
                        bootstrap_dt_bias=float(shortcut_cfg.get("bootstrap_dt_bias", 0.0)),
                        clip_intermediate=float(shortcut_cfg.get("clip_intermediate", 4.0)),
                    )
                elif objective in {"rectified_flow", "flow_matching", "naive"}:
                    loss, metrics = flow_matching_loss(model, z_l)
                else:
                    raise ValueError(f"unsupported macro objective: {objective}")
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
                for name in ("loss_flow", "loss_bootstrap", "bootstrap_ratio", "shortcut_dt_base_mean"):
                    if name in metrics:
                        logs[f"train/{name}"] = metrics[name].item()
                accelerator.log(logs, step=step)
                message = f"step={step} loss={logs['train/loss']:.6f} grad_norm={logs['train/grad_norm']:.4f}"
                if "train/loss_flow" in logs:
                    message += (
                        f" flow={logs['train/loss_flow']:.6f}"
                        f" bootstrap={logs['train/loss_bootstrap']:.6f}"
                    )
                accelerator.print(message)

            if fid_every > 0 and step % fid_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
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
                        fid_dataset,
                        vae,
                        fid_model,
                        num_samples=min(fid_num_samples, len(fid_dataset)),
                        batch_size=fid_batch_size,
                        sampler=fid_sampler,
                        sampler_steps=fid_sampler_steps,
                        device=accelerator.device,
                    )
                    if backup is not None:
                        with torch.no_grad():
                            for name, parameter in eval_model.named_parameters():
                                if name in backup:
                                    parameter.copy_(backup[name])
                    accelerator.log({"eval/fid_macro": fid}, step=step)
                    accelerator.print(f"step={step} fid_macro={fid:.4f}")
                    model.train()
                accelerator.wait_for_everyone()

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
