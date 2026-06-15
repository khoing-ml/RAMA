from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import math

import torch
import yaml
from accelerate import Accelerator
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.fid import FIDStats, InceptionFID, calculate_fid, stats_from_feature_batches
from src.dataset.latent_decomposition import reconstruct_from_decomposition, reconstruct_low_freq
from src.dataset.latent_dataset import CachedMicroLatentDataset
from src.micro.loss import categorical_micro_loss, categorical_micro_metrics, continuous_micro_nll_loss
from src.micro.micro_rama_categorical import build_categorical_micro_rama_net
from src.modules.micro_rama import build_context_encoder, build_micro_rama_net, sample_micro_latent
from src.modules.rama import make_orthogonal_bases, patchify, unpatchify
from src.dataset.vae import decode_latents, load_sd_vae
from src.rama.projector import RAMAProjector
from src.rama.tokenizer import RAMATokenizer, build_tokenizer_from_config, load_tokenizer_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the micro latent RAMA model.")
    parser.add_argument("--config", default="configs/debug_6gb_micro.yaml")
    parser.add_argument("--latents", default=None, help="Override latents.output_dir from config.")
    parser.add_argument("--out", default=None, help="Override output.dir from config.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-process batch size.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.total_steps for debug runs.")
    parser.add_argument("--micro-type", choices=("categorical", "continuous"), default=None)
    parser.add_argument("--tokenizer-config", default=None, help="Path to cache/rama_tokenizer_config.pt.")
    parser.add_argument("--bases", default=None, help="Override RAMA bases path.")
    parser.add_argument("--fid-every", type=int, default=None, help="Override evaluation.fid_every_steps; <=0 disables FID.")
    parser.add_argument("--fid-num-samples", type=int, default=None, help="Override evaluation.fid_num_samples.")
    parser.add_argument("--disable-wandb", action="store_true", help="Run without initializing Weights & Biases.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_derived_config(config: dict[str, object]) -> dict[str, object]:
    """Auto-compute all values that follow from micro_latent.patch_size and residual_shape."""
    micro_latent = config.setdefault("micro_latent", {})
    residual_shape = list(micro_latent.get("residual_shape", [4, 32, 32]))
    patch_size = int(micro_latent.get("patch_size", 2))
    C, H, W = residual_shape
    patch_grid = [H // patch_size, W // patch_size]
    num_patches = patch_grid[0] * patch_grid[1]
    patch_dim = int(micro_latent.get("patch_dim", C * patch_size * patch_size))

    micro_latent["patch_grid"] = patch_grid
    micro_latent["num_patches"] = num_patches

    rama_bases = config.setdefault("rama_bases", {})
    rama_bases["num_patches"] = num_patches
    rama_bases["patch_dim"] = patch_dim
    auto_cache = f"cache/rama_bases_p{num_patches}_d{patch_dim}.pt"
    rama_bases.setdefault("cache_path", auto_cache)

    rama = config.setdefault("rama", {})
    rama["patch_size"] = patch_size
    rama["num_patches"] = num_patches
    rama["patch_dim"] = patch_dim
    rama.setdefault("bases_path", rama_bases["cache_path"])

    context_encoder = config.setdefault("context_encoder", {})
    context_encoder["grid_size"] = patch_grid

    config.setdefault("micro", {})["patch_dim"] = patch_dim
    if config.get("micro_continuous"):
        config["micro_continuous"]["patch_dim"] = patch_dim

    return config


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
    context_encoder: torch.nn.Module,
    micro_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
    scheduler: object | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "context_encoder": context_encoder.state_dict(),
        "micro_model": micro_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)


def sample_tokens(logits: torch.Tensor, temperature: float, use_argmax: bool) -> torch.Tensor:
    if use_argmax:
        return logits.argmax(dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    return torch.multinomial(flat, num_samples=1).reshape(logits.shape[:-1])


@torch.no_grad()
def evaluate_micro_fid(
    context_encoder: torch.nn.Module,
    micro_model: torch.nn.Module,
    dataset: CachedMicroLatentDataset,
    vae: torch.nn.Module,
    fid_model: InceptionFID,
    projector: RAMAProjector,
    tokenizer: RAMATokenizer | None,
    micro_type: str,
    num_samples: int,
    batch_size: int,
    patch_size: int,
    temperature: float,
    use_argmax: bool,
    device: torch.device,
    real_stats: FIDStats | None = None,
) -> float:
    context_was_training = context_encoder.training
    micro_was_training = micro_model.training
    context_encoder.eval()
    micro_model.eval()
    real_batches: list[torch.Tensor] = []
    fake_batches: list[torch.Tensor] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    remaining = num_samples
    for batch in loader:
        if remaining <= 0:
            break
        z_l = batch["z_L"][:remaining].to(device)
        z_h = batch["z_H"][:remaining].to(device)
        if real_stats is None:
            z_real = reconstruct_from_decomposition(z_l, z_h)
            real_batches.append(fid_model(decode_latents(vae, z_real)))
        if micro_type == "categorical":
            if tokenizer is None:
                raise RuntimeError("categorical micro FID requires a RAMATokenizer")
            context = context_encoder(z_l)
            logits = micro_model(context)
            tokens = sample_tokens(logits, temperature=temperature, use_argmax=use_argmax)
            y_hat = tokenizer.dequantize(tokens)
            patches_hat = projector.inverse(y_hat)
            z_h_hat = unpatchify(
                patches_hat,
                channels=z_h.shape[1],
                height=z_h.shape[2],
                width=z_h.shape[3],
                patch_size=patch_size,
            )
        else:
            z_h_hat = sample_micro_latent(
                z_l,
                context_encoder,
                micro_model,
                projector.bases,
                latent_channels=z_h.shape[1],
                latent_height=z_h.shape[2],
                latent_width=z_h.shape[3],
                patch_size=patch_size,
            )
        z_fake = reconstruct_from_decomposition(z_l, z_h_hat)
        fake_batches.append(fid_model(decode_latents(vae, z_fake)))
        remaining -= z_l.shape[0]

    if context_was_training:
        context_encoder.train()
    if micro_was_training:
        micro_model.train()
    computed_real = real_stats if real_stats is not None else stats_from_feature_batches(real_batches)
    return calculate_fid(computed_real, stats_from_feature_batches(fake_batches))


@torch.no_grad()
def sample_and_save_images(
    context_encoder: torch.nn.Module,
    micro_model: torch.nn.Module,
    dataset: CachedMicroLatentDataset,
    vae: torch.nn.Module,
    projector: RAMAProjector,
    tokenizer: RAMATokenizer | None,
    micro_type: str,
    num_images: int,
    patch_size: int,
    temperature: float,
    use_argmax: bool,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    out_path: Path,
) -> torch.Tensor:
    """Save a comparison grid (macro-only | micro-reconstruction | ground-truth) and return it."""
    from torchvision.utils import make_grid, save_image

    context_was_training = context_encoder.training
    micro_was_training = micro_model.training
    context_encoder.eval()
    micro_model.eval()

    num_images = min(num_images, len(dataset))
    samples = [dataset[i] for i in range(num_images)]
    z_l = torch.stack([s["z_L"] for s in samples]).to(device)
    z_h = torch.stack([s["z_H"] for s in samples]).to(device)

    autocast_ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype else torch.autocast(device_type=device.type, enabled=False)
    with autocast_ctx:
        context = context_encoder(z_l)
        if micro_type == "categorical":
            if tokenizer is None:
                raise RuntimeError("categorical sampling requires a RAMATokenizer")
            logits = micro_model(context)
            tokens = sample_tokens(logits, temperature=temperature, use_argmax=use_argmax)
            y_hat = tokenizer.dequantize(tokens)
            patches_hat = projector.inverse(y_hat)
            z_h_hat = unpatchify(patches_hat, channels=z_h.shape[1], height=z_h.shape[2], width=z_h.shape[3], patch_size=patch_size)
        else:
            z_h_hat = sample_micro_latent(z_l, context_encoder, micro_model, projector.bases,
                                           latent_channels=z_h.shape[1], latent_height=z_h.shape[2],
                                           latent_width=z_h.shape[3], patch_size=patch_size)

    img_macro = decode_latents(vae, reconstruct_low_freq(z_l))
    img_micro = decode_latents(vae, reconstruct_from_decomposition(z_l, z_h_hat))
    img_real  = decode_latents(vae, reconstruct_from_decomposition(z_l, z_h))

    # interleave columns: macro | micro | real per row
    imgs = []
    for i in range(num_images):
        imgs.extend([img_macro[i], img_micro[i], img_real[i]])
    grid = make_grid(torch.stack(imgs), nrow=3, normalize=True, value_range=(-1, 1))
    macro_grid = make_grid(img_macro, nrow=max(1, int(num_images ** 0.5)), normalize=True, value_range=(-1, 1))
    micro_grid = make_grid(img_micro, nrow=max(1, int(num_images ** 0.5)), normalize=True, value_range=(-1, 1))
    real_grid  = make_grid(img_real,  nrow=max(1, int(num_images ** 0.5)), normalize=True, value_range=(-1, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)

    if context_was_training:
        context_encoder.train()
    if micro_was_training:
        micro_model.train()

    return grid, macro_grid, micro_grid, real_grid


def load_or_make_bases(config: dict[str, object], override_path: str | None = None) -> torch.Tensor:
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


def load_tokenizer(config: dict[str, object], override_path: str | None = None) -> RAMATokenizer:
    config_path = Path(str(override_path or config.get("config_path", "cache/rama_tokenizer_config.pt")))
    if config_path.exists():
        return build_tokenizer_from_config(load_tokenizer_config(str(config_path)))
    return RAMATokenizer(
        num_bins=int(config.get("num_bins", 256)),
        bound=float(config.get("bound", 3.0)),
    )


def _relaunch_with_accelerate(config: dict[str, object]) -> None:
    """Re-exec under accelerate launch using num_gpus and precision from config."""
    training = config.get("training", {})
    num_processes = int(training.get("num_gpus", 1))
    precision = str(training.get("precision", "no"))
    if precision not in {"fp16", "bf16"}:
        precision = "no"
    args = ["accelerate", "launch", f"--num_processes={num_processes}", f"--mixed_precision={precision}"]
    if num_processes > 1:
        args.append("--multi_gpu")
    os.environ["_RAMA_RELAUNCHED"] = "1"
    os.execvp("accelerate", args + sys.argv)


@torch.no_grad()
def evaluate_val_loss(
    context_encoder: torch.nn.Module,
    micro_model: torch.nn.Module,
    val_loader: DataLoader,
    projector: RAMAProjector,
    tokenizer: RAMATokenizer | None,
    micro_type: str,
    patch_size: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, float]:
    context_was_training = context_encoder.training
    micro_was_training = micro_model.training
    context_encoder.eval()
    micro_model.eval()

    total_loss = 0.0
    metric_accum: dict[str, float] = {}
    count = 0

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype
        else torch.autocast(device_type=device.type, enabled=False)
    )
    for batch in val_loader:
        z_l = batch["z_L"].to(device)
        z_h = batch["z_H"].to(device)
        with autocast_ctx:
            patches = patchify(z_h, patch_size=patch_size)
            y = projector.project(patches)
            context = context_encoder(z_l)
            if micro_type == "categorical":
                tokens = tokenizer.quantize(y)
                logits = micro_model(context)
                loss = categorical_micro_loss(logits, tokens, num_bins=tokenizer.num_bins)
                metrics = categorical_micro_metrics(logits, tokens, num_bins=tokenizer.num_bins)
                for k, v in metrics.items():
                    metric_accum[k] = metric_accum.get(k, 0.0) + v.float().item()
            else:
                eps, logabsdet = micro_model(y, context)
                loss = continuous_micro_nll_loss(eps, logabsdet)
                metric_accum["logabsdet_mean"] = (
                    metric_accum.get("logabsdet_mean", 0.0) + logabsdet.float().mean().item()
                )
        total_loss += loss.float().item()
        count += 1

    if context_was_training:
        context_encoder.train()
    if micro_was_training:
        micro_model.train()

    n = max(count, 1)
    result: dict[str, float] = {"val/loss": total_loss / n}
    for k, v in metric_accum.items():
        result[f"val/{k}"] = v / n
    return result


def main() -> None:
    if "LOCAL_RANK" not in os.environ and "_RAMA_RELAUNCHED" not in os.environ:
        _relaunch_with_accelerate(load_config(parse_args().config))
    args = parse_args()
    config = resolve_derived_config(load_config(args.config))
    training = config.get("training", {})
    logging_cfg = config.get("logging", {})
    evaluation_cfg = config.get("evaluation", {})
    micro_latent_cfg = config.get("micro_latent", {})
    tokenizer_cfg = config.get("tokenizer", {})

    mixed_precision = str(training.get("precision", "fp32"))
    if mixed_precision not in {"fp16", "bf16"}:
        mixed_precision = "no"

    tracker = None if args.disable_wandb else logging_cfg.get("tracker")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=mixed_precision,
        log_with="wandb" if tracker == "wandb" else None,
    )

    out_dir = Path(args.out or config.get("output", {}).get("dir", "outputs/micro_rama"))
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    latent_dir = args.latents or config.get("latents", {}).get("output_dir", "data/latents")
    full_dataset = CachedMicroLatentDataset(latent_dir)
    batch_size = args.batch_size or int(training.get("batch_size_per_gpu", 64))

    val_split = float(training.get("val_split", 0.05))
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    dataset = train_dataset

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
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    micro_type = args.micro_type or str(config.get("micro", {}).get("type", config.get("micro_rama_net", {}).get("type", "categorical")))
    if micro_type == "conditional_rq_nsf":
        micro_type = "continuous"
    if micro_type not in {"categorical", "continuous"}:
        raise ValueError(f"unsupported micro type: {micro_type}")
    config["micro_type"] = micro_type

    tokenizer = load_tokenizer(tokenizer_cfg, args.tokenizer_config) if micro_type == "categorical" else None
    patch_size = int(micro_latent_cfg.get("patch_size", 2))
    context_encoder_cfg = dict(config.get("context_encoder", {}))
    # CE patch_size is always micro patch_size // 2 because z_L is 2x downsampled from z_H
    context_encoder_cfg["patch_size"] = patch_size // 2
    context_encoder = build_context_encoder(context_encoder_cfg)
    if micro_type == "categorical":
        micro_model = build_categorical_micro_rama_net(
            config.get("micro", config.get("micro_rama_net", {})),
            num_bins=tokenizer.num_bins if tokenizer is not None else None,
        )
    else:
        micro_model = build_micro_rama_net(config.get("micro_continuous", config.get("micro_rama_net", {})))
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(micro_model.parameters()),
        lr=float(training.get("learning_rate", training.get("lr", 2.0e-4))),
        betas=tuple(training.get("betas", [0.9, 0.999])),
        weight_decay=float(training.get("weight_decay", 1.0e-4)),
    )
    rama_cfg = config.get("rama", config.get("rama_bases", {}))
    bases = load_or_make_bases(rama_cfg, args.bases)

    total_steps_for_sched = int(training.get("total_steps", 200000))
    warmup_steps = int(training.get("warmup_steps", total_steps_for_sched // 20))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(total_steps_for_sched - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    context_encoder, micro_model, optimizer, dataloader, scheduler = accelerator.prepare(
        context_encoder,
        micro_model,
        optimizer,
        dataloader,
        scheduler,
    )
    projector = RAMAProjector(bases).to(accelerator.device)
    projector.requires_grad_(False)

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        accelerator.unwrap_model(context_encoder).load_state_dict(checkpoint["context_encoder"])
        accelerator.unwrap_model(micro_model).load_state_dict(checkpoint["micro_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])

    if accelerator.is_main_process and tracker == "wandb":
        run_name = logging_cfg.get("run_name")
        accelerator.init_trackers(
            project_name=str(logging_cfg.get("project", "rama")),
            config=config,
            init_kwargs={"wandb": {"name": run_name}} if run_name else {},
        )

    context_noise_sigma = float(config.get("context_encoder", {}).get("context_noise_sigma", 0.03))
    grad_clip = float(training.get("grad_clip", 1.0))
    total_steps = args.max_steps or int(training.get("total_steps", 200000))
    log_every = int(logging_cfg.get("log_every_steps", 100))
    checkpoint_every = int(logging_cfg.get("checkpoint_every_steps", 10000))
    val_every = int(logging_cfg.get("val_every_steps", log_every * 5))
    fid_every = args.fid_every if args.fid_every is not None else int(evaluation_cfg.get("fid_every_steps", 0))
    fid_num_samples = args.fid_num_samples or int(evaluation_cfg.get("fid_num_samples", 512))
    fid_batch_size = int(evaluation_cfg.get("fid_batch_size", min(batch_size, 32)))
    fid_temperature = float(evaluation_cfg.get("temperature", 1.0))
    fid_use_argmax = bool(evaluation_cfg.get("use_argmax", False))
    sample_every = int(logging_cfg.get("sample_every_steps", 0))
    num_sample_images = int(logging_cfg.get("num_sample_images", 8))
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mixed_precision)
    precomputed_real_stats: FIDStats | None = None
    if accelerator.is_main_process and fid_every > 0:
        fid_stats_path = evaluation_cfg.get("fid_stats")
        if fid_stats_path:
            p = Path(fid_stats_path)
            if p.exists():
                data = torch.load(p, map_location="cpu")
                precomputed_real_stats = FIDStats(
                    mean=data["mean"], covariance=data["covariance"], num_samples=data["num_samples"]
                )
                accelerator.print(f"Loaded precomputed real FID stats from {p} ({precomputed_real_stats.num_samples} samples)")
            else:
                accelerator.print(f"WARNING: evaluation.fid_stats path {fid_stats_path!r} not found; will compute real stats each eval")

    vae = None
    fid_model = None
    if accelerator.is_main_process and (fid_every > 0 or sample_every > 0):
        vae_cfg = config.get("vae", {})
        vae = load_sd_vae(
            checkpoint=resolve_vae_checkpoint(vae_cfg),
            cache_dir=str(vae_cfg.get("cache_dir", ".cache/huggingface")),
            dtype=str(vae_cfg.get("dtype", "fp16")),
            device=str(accelerator.device),
        )
    if accelerator.is_main_process and fid_every > 0:
        fid_model = InceptionFID(accelerator.device)

    step = start_step
    context_encoder.train()
    micro_model.train()
    loss_accum = 0.0
    metric_accum: dict[str, float] = {}
    metric_count = 0

    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break
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
                    loss = categorical_micro_loss(logits, tokens, num_bins=tokenizer.num_bins)
                    with torch.no_grad():
                        token_metrics = categorical_micro_metrics(logits, tokens, num_bins=tokenizer.num_bins)
                else:
                    eps, logabsdet = micro_model(y, context)
                    loss = continuous_micro_nll_loss(eps, logabsdet)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        list(context_encoder.parameters()) + list(micro_model.parameters()),
                        grad_clip,
                    )
                else:
                    grad_norm = torch.tensor(0.0, device=accelerator.device)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            loss_accum += loss.detach().float().item()
            metric_count += 1
            if micro_type == "categorical":
                for name, value in token_metrics.items():
                    metric_accum[name] = metric_accum.get(name, 0.0) + value.detach().float().item()
            else:
                metric_accum["logabsdet_mean"] = (
                    metric_accum.get("logabsdet_mean", 0.0) + logabsdet.detach().float().mean().item()
                )
            if step % log_every == 0:
                loss_mean = loss_accum / max(metric_count, 1)
                logs = {
                    "train/loss": loss_mean,
                    "train/loss_last": loss.detach().float().item(),
                    "train/grad_norm": float(grad_norm),
                    "train/y_abs_mean": y.detach().abs().float().mean().item(),
                    "train/lr": scheduler.get_last_lr()[0],
                }
                if micro_type == "categorical":
                    for name in sorted(metric_accum):
                        logs[f"train/{name}"] = metric_accum[name] / max(metric_count, 1)
                else:
                    logs["train/logabsdet_mean"] = metric_accum["logabsdet_mean"] / max(metric_count, 1)
                accelerator.log(logs, step=step)
                message = (
                    f"step={step} loss={logs['train/loss']:.6f} "
                    f"grad_norm={logs['train/grad_norm']:.4f} "
                    f"y_abs_mean={logs['train/y_abs_mean']:.4f}"
                )
                if micro_type == "categorical":
                    message += (
                        f" token_acc={logs['train/token_acc']:.4f} "
                        f"top5={logs['train/token_top5_acc']:.4f} "
                        f"within1={logs['train/token_within_1']:.4f}"
                    )
                else:
                    message += f" logabsdet_mean={logs['train/logabsdet_mean']:.4f}"
                accelerator.print(message)
                loss_accum = 0.0
                metric_accum.clear()
                metric_count = 0

            if sample_every > 0 and step % sample_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    out_path = out_dir / "samples" / f"step_{step:08d}.png"
                    grid = sample_and_save_images(
                        accelerator.unwrap_model(context_encoder),
                        accelerator.unwrap_model(micro_model),
                        dataset, vae, projector, tokenizer, micro_type,
                        num_images=num_sample_images,
                        patch_size=patch_size,
                        temperature=fid_temperature,
                        use_argmax=fid_use_argmax,
                        device=accelerator.device,
                        autocast_dtype=autocast_dtype,
                        out_path=out_path,
                    )
                    if tracker == "wandb":
                        import wandb
                        grid, macro_grid, micro_grid, real_grid = grid
                        accelerator.log({
                            "samples/comparison": wandb.Image(grid),
                            "samples/macro_grid": wandb.Image(macro_grid),
                            "samples/micro_grid": wandb.Image(micro_grid),
                            "samples/real_grid":  wandb.Image(real_grid),
                        }, step=step)
                    accelerator.print(f"step={step} samples saved to {out_path}")
                    context_encoder.train()
                    micro_model.train()
                accelerator.wait_for_everyone()

            if val_every > 0 and step % val_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    val_metrics = evaluate_val_loss(
                        accelerator.unwrap_model(context_encoder),
                        accelerator.unwrap_model(micro_model),
                        val_loader,
                        projector,
                        tokenizer,
                        micro_type,
                        patch_size=patch_size,
                        device=accelerator.device,
                        autocast_dtype=autocast_dtype,
                    )
                    accelerator.log(val_metrics, step=step)
                    val_loss_str = f"val/loss={val_metrics['val/loss']:.6f}"
                    if micro_type == "categorical":
                        val_loss_str += (
                            f" val/token_acc={val_metrics.get('val/token_acc', 0):.4f}"
                            f" val/token_within_1={val_metrics.get('val/token_within_1', 0):.4f}"
                        )
                    accelerator.print(f"step={step} {val_loss_str}")
                    context_encoder.train()
                    micro_model.train()
                accelerator.wait_for_everyone()

            if fid_every > 0 and step % fid_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    fid = evaluate_micro_fid(
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
                        patch_size=patch_size,
                        temperature=fid_temperature,
                        use_argmax=fid_use_argmax,
                        device=accelerator.device,
                        real_stats=precomputed_real_stats,
                    )
                    accelerator.log({"eval/fid_micro_real_zL": fid}, step=step)
                    accelerator.print(f"step={step} fid_micro_real_zL={fid:.4f}")
                    context_encoder.train()
                    micro_model.train()
                accelerator.wait_for_everyone()

            if accelerator.is_main_process and step % checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_path(out_dir, step),
                    step,
                    accelerator.unwrap_model(context_encoder),
                    accelerator.unwrap_model(micro_model),
                    optimizer,
                    copy.deepcopy(config),
                    scheduler=scheduler,
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
            scheduler=scheduler,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
