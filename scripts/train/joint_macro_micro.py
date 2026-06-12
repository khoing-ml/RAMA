from __future__ import annotations

import argparse
import copy
import sys
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.macro_flow import (
    checkpoint_path as macro_checkpoint_path,
    resolve_vae_checkpoint,
    save_checkpoint as save_macro_checkpoint,
)
from scripts.train.micro_rama import (
    checkpoint_path as micro_checkpoint_path,
    load_or_make_bases,
    load_tokenizer,
    sample_tokens,
    save_checkpoint as save_micro_checkpoint,
)
from src.evaluation.fid import InceptionFID, calculate_fid, stats_from_feature_batches
from src.macro.sampler import sample_macro_latents
from src.micro.loss import categorical_micro_loss, categorical_micro_metrics, continuous_micro_nll_loss
from src.micro.micro_rama_categorical import build_categorical_micro_rama_net
from src.modules.ema import EMA
from src.macro.losses import flow_matching_loss
from src.data.latent_decomposition import reconstruct_from_decomposition
from src.data.latent_dataset import CachedMicroLatentDataset
from src.macro.factory import build_macro_flow_model
from src.modules.micro_rama import build_context_encoder, build_micro_rama_net, sample_micro_latent
from src.modules.rama import patchify, unpatchify
from src.data.vae import decode_latents, load_sd_vae
from src.rama.projector import RAMAProjector


def log_sample_grids(samples_dir: Path, joint_step: int, accelerator: Accelerator) -> None:
    try:
        import wandb
    except ImportError:
        accelerator.print("wandb is not installed; sample grids were saved but not logged")
        return

    step_dir = samples_dir / f"step_{joint_step:08d}"
    image_paths = {
        "samples/comparison_grid": step_dir / "comparison.png",
        "samples/full_grid": step_dir / "full.png",
        "samples/macro_grid": step_dir / "macro.png",
        "samples/micro_grid": step_dir / "micro.png",
    }
    logs = {
        name: wandb.Image(str(path), caption=f"{name} step {joint_step}")
        for name, path in image_paths.items()
        if path.exists()
    }
    if logs:
        accelerator.log(logs, step=joint_step)


@torch.no_grad()
def evaluate_full_fid(
    macro_model: torch.nn.Module,
    context_encoder: torch.nn.Module,
    micro_model: torch.nn.Module,
    dataset: CachedMicroLatentDataset,
    vae: torch.nn.Module,
    fid_model: InceptionFID,
    projector: RAMAProjector,
    tokenizer,
    micro_type: str,
    num_samples: int,
    batch_size: int,
    sampler: str,
    sampler_steps: int,
    patch_size: int,
    latent_channels: int,
    latent_height: int,
    latent_width: int,
    temperature: float,
    use_argmax: bool,
    noise_scale: float,
    device: torch.device,
) -> float:
    macro_was_training = macro_model.training
    ctx_was_training = context_encoder.training
    micro_was_training = micro_model.training
    macro_model.eval()
    context_encoder.eval()
    micro_model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    real_batches: list[torch.Tensor] = []
    fake_batches: list[torch.Tensor] = []

    remaining = num_samples
    for batch in loader:
        if remaining <= 0:
            break
        z_l = batch["z_L"][:remaining].to(device)
        z_h = batch["z_H"][:remaining].to(device)
        z_real = reconstruct_from_decomposition(z_l, z_h)
        real_batches.append(fid_model(decode_latents(vae, z_real)))
        remaining -= z_l.shape[0]

    remaining = num_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        shape = (current_batch, macro_model.in_channels, macro_model.resolution, macro_model.resolution)
        z_l = sample_macro_latents(macro_model, shape=shape, method=sampler, num_steps=sampler_steps, device=str(device))
        z_l_up = F.interpolate(z_l, size=(latent_height, latent_width), mode="bilinear", align_corners=False)
        if micro_type == "categorical":
            if tokenizer is None:
                raise RuntimeError("categorical full FID requires a RAMATokenizer")
            context = context_encoder(z_l)
            logits = micro_model(context)
            tokens = sample_tokens(logits, temperature=temperature, use_argmax=use_argmax)
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
                noise_scale=noise_scale,
            )
        z_hat = z_l_up + z_h_hat
        fake_batches.append(fid_model(decode_latents(vae, z_hat)))
        remaining -= current_batch

    if macro_was_training:
        macro_model.train()
    if ctx_was_training:
        context_encoder.train()
    if micro_was_training:
        micro_model.train()
    return calculate_fid(stats_from_feature_batches(real_batches), stats_from_feature_batches(fake_batches))


@torch.no_grad()
def sample_and_save(
    macro_model: torch.nn.Module,
    context_encoder: torch.nn.Module,
    micro_model: torch.nn.Module,
    vae: torch.nn.Module,
    projector: RAMAProjector,
    tokenizer,
    micro_type: str,
    samples_dir: Path,
    joint_step: int,
    num_samples: int,
    sampler: str,
    sampler_steps: int,
    patch_size: int,
    latent_channels: int,
    latent_height: int,
    latent_width: int,
    temperature: float,
    use_argmax: bool,
    noise_scale: float,
    device: torch.device,
) -> None:
    macro_was_training = macro_model.training
    ctx_was_training = context_encoder.training
    micro_was_training = micro_model.training
    macro_model.eval()
    context_encoder.eval()
    micro_model.eval()

    shape = (num_samples, macro_model.in_channels, macro_model.resolution, macro_model.resolution)
    z_l = sample_macro_latents(macro_model, shape=shape, method=sampler, num_steps=sampler_steps, device=str(device))
    z_l_up = F.interpolate(z_l, size=(latent_height, latent_width), mode="bilinear", align_corners=False)
    if micro_type == "categorical":
        if tokenizer is None:
            raise RuntimeError("categorical sampling requires a RAMATokenizer")
        context = context_encoder(z_l)
        logits = micro_model(context)
        tokens = sample_tokens(logits, temperature=temperature, use_argmax=use_argmax)
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
            noise_scale=noise_scale,
        )
    z_hat = z_l_up + z_h_hat

    macro_images = decode_latents(vae, z_l_up)
    micro_images = decode_latents(vae, z_h_hat)
    full_images = decode_latents(vae, z_hat)

    macro_images = macro_images.clamp(-1, 1).mul(0.5).add(0.5)
    micro_images = micro_images.clamp(-1, 1).mul(0.5).add(0.5)
    full_images = full_images.clamp(-1, 1).mul(0.5).add(0.5)

    nrow = max(1, int(num_samples ** 0.5))
    step_dir = samples_dir / f"step_{joint_step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    save_image(macro_images, step_dir / "macro.png", nrow=nrow)
    save_image(micro_images, step_dir / "micro.png", nrow=nrow)
    save_image(full_images, step_dir / "full.png", nrow=nrow)

    comparison = torch.stack([macro_images, micro_images, full_images], dim=1).flatten(0, 1)
    save_image(comparison, step_dir / "comparison.png", nrow=3)

    if macro_was_training:
        macro_model.train()
    if ctx_was_training:
        context_encoder.train()
    if micro_was_training:
        micro_model.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train macro flow and micro RAMA in one two-stage step loop.")
    parser.add_argument("--macro-config", default="configs/celeba256_sdvae_macro.yaml")
    parser.add_argument("--micro-config", default="configs/celeba256_sdvae_micro.yaml")
    parser.add_argument("--latents", default=None)
    parser.add_argument("--macro-out", default="outputs/macro_flow")
    parser.add_argument("--micro-out", default="outputs/micro_rama")
    parser.add_argument("--macro-resume", default=None)
    parser.add_argument("--micro-resume", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--macro-batch-size", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--macro-max-steps", type=int, default=None)
    parser.add_argument("--micro-max-steps", type=int, default=None)
    parser.add_argument("--micro-type", choices=("categorical", "continuous"), default=None)
    parser.add_argument("--tokenizer-config", default=None)
    parser.add_argument("--bases", default=None)
    parser.add_argument("--fid-every", type=int, default=None)
    parser.add_argument("--fid-num-samples", type=int, default=None)
    parser.add_argument("--fid-batch-size", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sampler", choices=("heun", "euler"), default="heun")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-argmax", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Gaussian base-noise scale for continuous micro RAMA.")
    parser.add_argument("--samples-dir", default="outputs/samples")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dataloader_cycle(dataloader: DataLoader):
    for batch in cycle(dataloader):
        yield batch


def main() -> None:
    args = parse_args()
    macro_config = load_config(args.macro_config)
    micro_config = load_config(args.micro_config)
    macro_training = macro_config.get("training", {})
    micro_training = micro_config.get("training", {})
    macro_logging = macro_config.get("logging", {})
    micro_logging = micro_config.get("logging", {})
    macro_eval = macro_config.get("evaluation", {})
    micro_eval = micro_config.get("evaluation", {})
    micro_latent_cfg = micro_config.get("micro_latent", {})

    mixed_precision = str(macro_training.get("precision", micro_training.get("precision", "fp32")))
    if mixed_precision not in {"fp16", "bf16"}:
        mixed_precision = "no"

    tracker = None if args.disable_wandb else macro_logging.get("tracker", micro_logging.get("tracker"))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(macro_training.get("gradient_accumulation_steps", 1)),
        mixed_precision=mixed_precision,
        log_with="wandb" if tracker == "wandb" else None,
    )

    macro_out = Path(args.macro_out)
    micro_out = Path(args.micro_out)
    if accelerator.is_main_process:
        macro_out.mkdir(parents=True, exist_ok=True)
        micro_out.mkdir(parents=True, exist_ok=True)

    latent_dir = args.latents or micro_config.get("latents", {}).get(
        "output_dir",
        macro_config.get("latents", {}).get("output_dir", "data/latents"),
    )
    dataset = CachedMicroLatentDataset(latent_dir)
    macro_batch_size = args.macro_batch_size or int(macro_training.get("batch_size_per_gpu", 64))
    micro_batch_size = args.micro_batch_size or int(micro_training.get("batch_size_per_gpu", 64))
    macro_loader = DataLoader(
        dataset,
        batch_size=macro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    micro_loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if len(macro_loader) == 0:
        raise ValueError("latent dataset is smaller than the macro per-process batch size")
    if len(micro_loader) == 0:
        raise ValueError("latent dataset is smaller than the micro per-process batch size")

    macro_model = build_macro_flow_model(macro_config.get("macro_flow_model", {}))
    macro_optimizer = torch.optim.AdamW(
        macro_model.parameters(),
        lr=float(macro_training.get("learning_rate", macro_training.get("lr", 2.0e-4))),
        weight_decay=float(macro_training.get("weight_decay", 0.0)),
    )
    macro_ema = EMA(macro_model, float(macro_training.get("ema_decay", 0.9999))) if macro_training.get("ema", True) else None

    micro_type = args.micro_type or str(
        micro_config.get("micro", {}).get("type", micro_config.get("micro_rama_net", {}).get("type", "categorical"))
    )
    if micro_type == "conditional_rq_nsf":
        micro_type = "continuous"
    if micro_type not in {"categorical", "continuous"}:
        raise ValueError(f"unsupported micro type: {micro_type}")
    micro_config["micro_type"] = micro_type

    tokenizer = load_tokenizer(micro_config.get("tokenizer", {}), args.tokenizer_config) if micro_type == "categorical" else None
    context_encoder = build_context_encoder(micro_config.get("context_encoder", {}))
    if micro_type == "categorical":
        micro_model = build_categorical_micro_rama_net(
            micro_config.get("micro", micro_config.get("micro_rama_net", {})),
            num_bins=tokenizer.num_bins if tokenizer is not None else None,
        )
    else:
        micro_model = build_micro_rama_net(micro_config.get("micro_continuous", micro_config.get("micro_rama_net", {})))
    micro_optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(micro_model.parameters()),
        lr=float(micro_training.get("learning_rate", micro_training.get("lr", 2.0e-4))),
        betas=tuple(micro_training.get("betas", [0.9, 0.999])),
        weight_decay=float(micro_training.get("weight_decay", 1.0e-4)),
    )
    bases = load_or_make_bases(micro_config.get("rama", micro_config.get("rama_bases", {})), args.bases)

    (
        macro_model,
        macro_optimizer,
        context_encoder,
        micro_model,
        micro_optimizer,
        macro_loader,
        micro_loader,
    ) = accelerator.prepare(
        macro_model,
        macro_optimizer,
        context_encoder,
        micro_model,
        micro_optimizer,
        macro_loader,
        micro_loader,
    )
    projector = RAMAProjector(bases).to(accelerator.device)
    projector.requires_grad_(False)

    macro_start_step = 0
    if args.macro_resume:
        checkpoint = torch.load(args.macro_resume, map_location="cpu")
        accelerator.unwrap_model(macro_model).load_state_dict(checkpoint["model"])
        macro_optimizer.load_state_dict(checkpoint["optimizer"])
        if macro_ema is not None and checkpoint.get("ema") is not None:
            macro_ema.load_state_dict(checkpoint["ema"])
        macro_start_step = int(checkpoint["step"])

    micro_start_step = 0
    if args.micro_resume:
        checkpoint = torch.load(args.micro_resume, map_location="cpu")
        accelerator.unwrap_model(context_encoder).load_state_dict(checkpoint["context_encoder"])
        accelerator.unwrap_model(micro_model).load_state_dict(checkpoint["micro_model"])
        micro_optimizer.load_state_dict(checkpoint["optimizer"])
        micro_start_step = int(checkpoint["step"])

    if accelerator.is_main_process and tracker == "wandb":
        accelerator.init_trackers(
            project_name=str(macro_logging.get("project", micro_logging.get("project", "rama"))),
            config={"macro": macro_config, "micro": micro_config, "joint_training": vars(args)},
            init_kwargs={
                "wandb": {
                    "entity": macro_logging.get("entity", micro_logging.get("entity")),
                    "name": str(macro_logging.get("run_name", "macro")) + "+micro-joint",
                }
            },
        )

    macro_total_steps = args.macro_max_steps or int(macro_training.get("total_steps", macro_training.get("max_steps", 200000)))
    micro_total_steps = args.micro_max_steps or int(micro_training.get("total_steps", micro_training.get("max_steps", 200000)))
    macro_log_every = int(macro_logging.get("log_every_steps", 100))
    micro_log_every = int(micro_logging.get("log_every_steps", 100))
    checkpoint_every = min(
        int(macro_logging.get("checkpoint_every_steps", 10000)),
        int(micro_logging.get("checkpoint_every_steps", 10000)),
    )
    fid_every = args.fid_every if args.fid_every is not None else int(macro_eval.get("fid_every_steps", 0))
    fid_num_samples = args.fid_num_samples or int(macro_eval.get("fid_num_samples", 512))
    fid_batch_size = args.fid_batch_size or int(macro_eval.get("fid_batch_size", min(macro_batch_size, 32)))
    sample_every = args.sample_every if args.sample_every is not None else int(macro_logging.get("sample_every_steps", 0))
    num_samples = args.num_samples
    fid_sampler = args.sampler
    fid_sampler_steps = args.sample_steps
    fid_temperature = args.temperature
    fid_use_argmax = args.sample_argmax or bool(micro_eval.get("use_argmax", False))
    fid_noise_scale = float(micro_eval.get("noise_scale", args.noise_scale))
    macro_full_latent_size = int(macro_eval.get("full_latent_size", 32))
    samples_dir = Path(args.samples_dir)
    macro_grad_clip = float(macro_training.get("grad_clip", 1.0))
    micro_grad_clip = float(micro_training.get("grad_clip", 1.0))
    patch_size = int(micro_latent_cfg.get("patch_size", 2))
    context_noise_sigma = float(micro_config.get("context_encoder", {}).get("context_noise_sigma", 0.03))
    residual_shape = [int(v) for v in micro_latent_cfg.get("residual_shape", [4, macro_full_latent_size, macro_full_latent_size])]
    latent_channels, latent_height, latent_width = residual_shape

    vae = None
    fid_model = None
    if accelerator.is_main_process and (fid_every > 0 or sample_every > 0):
        vae_cfg = macro_config.get("vae", micro_config.get("vae", {}))
        vae = load_sd_vae(
            checkpoint=resolve_vae_checkpoint(vae_cfg),
            cache_dir=str(vae_cfg.get("cache_dir", ".cache/huggingface")),
            dtype=str(vae_cfg.get("dtype", "fp16")),
            device=str(accelerator.device),
        )
        fid_model = InceptionFID(accelerator.device)

    macro_step = macro_start_step
    micro_step = micro_start_step
    joint_step = max(macro_step, micro_step)
    macro_batches = dataloader_cycle(macro_loader)
    micro_batches = dataloader_cycle(micro_loader)
    macro_model.train()
    context_encoder.train()
    micro_model.train()
    micro_loss_accum = 0.0
    micro_metric_accum: dict[str, float] = {}
    micro_metric_count = 0

    while macro_step < macro_total_steps or micro_step < micro_total_steps:
        if macro_step < macro_total_steps:
            batch = next(macro_batches)
            z_l = batch["z_L"].detach()
            with accelerator.accumulate(macro_model):
                macro_loss, macro_metrics = flow_matching_loss(macro_model, z_l)
                accelerator.backward(macro_loss)
                if accelerator.sync_gradients:
                    macro_grad_norm = accelerator.clip_grad_norm_(macro_model.parameters(), macro_grad_clip)
                else:
                    macro_grad_norm = torch.tensor(0.0, device=accelerator.device)
                macro_optimizer.step()
                macro_optimizer.zero_grad(set_to_none=True)
                if macro_ema is not None and accelerator.sync_gradients:
                    macro_ema.update(accelerator.unwrap_model(macro_model))
            macro_step += 1

            if macro_step % macro_log_every == 0:
                current_step = max(macro_step, micro_step)
                logs = {
                    "macro/train/loss": macro_metrics["loss"].item(),
                    "macro/train/target_v_norm": macro_metrics["target_v_norm"].item(),
                    "macro/train/pred_v_norm": macro_metrics["pred_v_norm"].item(),
                    "macro/train/grad_norm": float(macro_grad_norm),
                    "macro/train/lr": macro_optimizer.param_groups[0]["lr"],
                }
                accelerator.log(logs, step=current_step)
                accelerator.print(
                    f"macro_step={macro_step} loss={logs['macro/train/loss']:.6f} "
                    f"grad_norm={logs['macro/train/grad_norm']:.4f}"
                )

            if accelerator.is_main_process and macro_step % checkpoint_every == 0:
                save_macro_checkpoint(
                    macro_checkpoint_path(macro_out, macro_step),
                    macro_step,
                    accelerator.unwrap_model(macro_model),
                    macro_ema,
                    macro_optimizer,
                    copy.deepcopy(macro_config),
                )

        if micro_step < micro_total_steps:
            batch = next(micro_batches)
            z_l = batch["z_L"].detach()
            z_h = batch["z_H"].detach()
            with accelerator.accumulate(micro_model):
                z_l_input = z_l
                if context_noise_sigma > 0:
                    z_l_input = z_l_input + context_noise_sigma * torch.randn_like(z_l_input)
                patches = patchify(z_h, patch_size=patch_size)
                y = projector.project(patches)
                context = context_encoder(z_l_input)
                if micro_type == "categorical":
                    if tokenizer is None:
                        raise RuntimeError("categorical micro training requires a RAMATokenizer")
                    tokens = tokenizer.quantize(y)
                    logits = micro_model(context)
                    micro_loss = categorical_micro_loss(logits, tokens, num_bins=tokenizer.num_bins)
                    with torch.no_grad():
                        token_metrics = categorical_micro_metrics(logits, tokens, num_bins=tokenizer.num_bins)
                else:
                    eps, logabsdet = micro_model(y, context)
                    micro_loss = continuous_micro_nll_loss(eps, logabsdet)
                accelerator.backward(micro_loss)
                if accelerator.sync_gradients:
                    micro_grad_norm = accelerator.clip_grad_norm_(
                        list(context_encoder.parameters()) + list(micro_model.parameters()),
                        micro_grad_clip,
                    )
                else:
                    micro_grad_norm = torch.tensor(0.0, device=accelerator.device)
                micro_optimizer.step()
                micro_optimizer.zero_grad(set_to_none=True)
            micro_step += 1
            micro_loss_accum += micro_loss.detach().float().item()
            micro_metric_count += 1
            if micro_type == "categorical":
                for name, value in token_metrics.items():
                    micro_metric_accum[name] = micro_metric_accum.get(name, 0.0) + value.detach().float().item()
            else:
                micro_metric_accum["logabsdet_mean"] = (
                    micro_metric_accum.get("logabsdet_mean", 0.0) + logabsdet.detach().float().mean().item()
                )

            if micro_step % micro_log_every == 0:
                current_step = max(macro_step, micro_step)
                micro_loss_mean = micro_loss_accum / max(micro_metric_count, 1)
                logs = {
                    "micro/train/loss": micro_loss_mean,
                    "micro/train/loss_last": micro_loss.detach().float().item(),
                    "micro/train/grad_norm": float(micro_grad_norm),
                    "micro/train/y_abs_mean": y.detach().abs().float().mean().item(),
                    "micro/train/lr": micro_optimizer.param_groups[0]["lr"],
                }
                if micro_type == "categorical":
                    for name in sorted(micro_metric_accum):
                        logs[f"micro/train/{name}"] = micro_metric_accum[name] / max(micro_metric_count, 1)
                else:
                    logs["micro/train/logabsdet_mean"] = (
                        micro_metric_accum["logabsdet_mean"] / max(micro_metric_count, 1)
                    )
                accelerator.log(logs, step=current_step)
                message = (
                    f"micro_step={micro_step} loss={logs['micro/train/loss']:.6f} "
                    f"grad_norm={logs['micro/train/grad_norm']:.4f} "
                    f"y_abs_mean={logs['micro/train/y_abs_mean']:.4f}"
                )
                if micro_type == "categorical":
                    message += (
                        f" token_acc={logs['micro/train/token_acc']:.4f} "
                        f"top5={logs['micro/train/token_top5_acc']:.4f} "
                        f"within1={logs['micro/train/token_within_1']:.4f}"
                    )
                else:
                    message += f" logabsdet_mean={logs['micro/train/logabsdet_mean']:.4f}"
                accelerator.print(message)
                micro_loss_accum = 0.0
                micro_metric_accum.clear()
                micro_metric_count = 0

            if accelerator.is_main_process and micro_step % checkpoint_every == 0:
                save_micro_checkpoint(
                    micro_checkpoint_path(micro_out, micro_step),
                    micro_step,
                    accelerator.unwrap_model(context_encoder),
                    accelerator.unwrap_model(micro_model),
                    micro_optimizer,
                    copy.deepcopy(micro_config),
                )

        joint_step = max(macro_step, micro_step)

        if fid_every > 0 and joint_step > 0 and joint_step % fid_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                eval_model = accelerator.unwrap_model(macro_model)
                backup = None
                if macro_ema is not None:
                    backup = {
                        name: parameter.detach().clone()
                        for name, parameter in eval_model.named_parameters()
                        if parameter.requires_grad
                    }
                    macro_ema.copy_to(eval_model)
                fid = evaluate_full_fid(
                    eval_model,
                    accelerator.unwrap_model(context_encoder),
                    accelerator.unwrap_model(micro_model),
                    dataset,
                    vae,
                    fid_model,
                    projector,
                    tokenizer,
                    micro_type,
                    num_samples=min(fid_num_samples, len(dataset)),
                    batch_size=fid_batch_size,
                    sampler=fid_sampler,
                    sampler_steps=fid_sampler_steps,
                    patch_size=patch_size,
                    latent_channels=latent_channels,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    temperature=fid_temperature,
                    use_argmax=fid_use_argmax,
                    noise_scale=fid_noise_scale,
                    device=accelerator.device,
                )
                if backup is not None:
                    with torch.no_grad():
                        for name, parameter in eval_model.named_parameters():
                            if name in backup:
                                parameter.copy_(backup[name])
                accelerator.log({"eval/fid_full": fid}, step=joint_step)
                accelerator.print(f"step={joint_step} fid_full={fid:.4f}")
                macro_model.train()
                context_encoder.train()
                micro_model.train()
            accelerator.wait_for_everyone()

        if sample_every > 0 and joint_step > 0 and joint_step % sample_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                sample_and_save(
                    accelerator.unwrap_model(macro_model),
                    accelerator.unwrap_model(context_encoder),
                    accelerator.unwrap_model(micro_model),
                    vae,
                    projector,
                    tokenizer,
                    micro_type,
                    samples_dir,
                    joint_step,
                    num_samples=num_samples,
                    sampler=fid_sampler,
                    sampler_steps=fid_sampler_steps,
                    patch_size=patch_size,
                    latent_channels=latent_channels,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    temperature=fid_temperature,
                    use_argmax=fid_use_argmax,
                    noise_scale=fid_noise_scale,
                    device=accelerator.device,
                )
                if tracker == "wandb":
                    log_sample_grids(samples_dir, joint_step, accelerator)
                accelerator.print(f"step={joint_step} saved samples to {samples_dir}")
            accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_macro_checkpoint(
            macro_checkpoint_path(macro_out, macro_step),
            macro_step,
            accelerator.unwrap_model(macro_model),
            macro_ema,
            macro_optimizer,
            copy.deepcopy(macro_config),
        )
        save_micro_checkpoint(
            micro_checkpoint_path(micro_out, micro_step),
            micro_step,
            accelerator.unwrap_model(context_encoder),
            accelerator.unwrap_model(micro_model),
            micro_optimizer,
            copy.deepcopy(micro_config),
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
