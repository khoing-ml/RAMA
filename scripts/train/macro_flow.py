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
from src.evaluation.fid import FIDStats, InceptionFID, calculate_fid, stats_from_feature_batches
from src.macro.losses import flow_matching_loss, shortcut_matching_loss
from src.macro.sampler import sample_macro_latents
from src.dataset.latent_dataset import CachedLatentDataset, CachedMicroLatentDataset
from src.dataset.latent_decomposition import reconstruct_from_decomposition, reconstruct_low_freq
from src.macro.factory import build_macro_flow_model
from src.dataset.vae import decode_latents, load_sd_vae


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


def load_precomputed_fid_stats(path: str | Path) -> FIDStats:
    """Load pre-computed real FID statistics from a .npz (ADM/shortcut style) or .pt file."""
    path = Path(path)
    if path.suffix == ".npz":
        import numpy as np
        data = np.load(str(path))
        mean_key = next((k for k in ("mu", "mean", "m") if k in data), None)
        cov_key = next((k for k in ("sigma", "cov", "covariance") if k in data), None)
        if mean_key is None or cov_key is None:
            raise KeyError(f"cannot find mean/cov in {path}; available keys: {list(data.keys())}")
        mean = torch.tensor(data[mean_key]).double()
        cov = torch.tensor(data[cov_key]).double()
        n = int(data.get("n", data.get("num_samples", 0)))
    else:
        data = torch.load(str(path), map_location="cpu")
        if isinstance(data, FIDStats):
            return data
        mean = data["mean"].double()
        cov = data["covariance"].double()
        n = int(data.get("num_samples", 0))
    return FIDStats(mean=mean, covariance=cov, num_samples=n)


@torch.no_grad()
def evaluate_multi_step_fid(
    model: torch.nn.Module,
    fid_dataset: CachedMicroLatentDataset,
    vae: torch.nn.Module,
    fid_model: InceptionFID,
    num_samples: int,
    batch_size: int,
    sampler: str,
    sampler_steps_list: list[int],
    device: torch.device,
    real_stats: FIDStats | None = None,
    num_sample_images: int = 16,
) -> tuple[dict[int, float], dict[int, dict[str, torch.Tensor]]]:
    """Evaluate FID and collect sample images for each step count in sampler_steps_list.

    If real_stats is provided (pre-computed), skips real Inception forward passes.
    Returns (fid_by_steps, components_by_steps) where components_by_steps maps each step
    count to a dict with keys "output", "low_freq", and "z_h".
    """
    was_training = model.training
    model.eval()

    loader = DataLoader(fid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    remaining = num_samples
    compute_real = real_stats is None
    real_feat_batches: list[torch.Tensor] = []
    z_h_store: list[torch.Tensor] = []
    shapes: list[tuple[int, int, int, int]] = []

    for batch in loader:
        if remaining <= 0:
            break
        z_l = batch["z_L"][:remaining].to(device)
        z_h = batch["z_H"][:remaining].to(device)
        if compute_real:
            z_real = reconstruct_from_decomposition(z_l, z_h)
            real_feat_batches.append(fid_model(decode_latents(vae, z_real)))
        z_h_store.append(z_h.cpu())
        shapes.append((z_l.shape[0], model.in_channels, model.resolution, model.resolution))
        remaining -= z_l.shape[0]

    if compute_real:
        real_stats = stats_from_feature_batches(real_feat_batches)

    fid_results: dict[int, float] = {}
    sample_components: dict[int, dict[str, torch.Tensor]] = {}

    for steps in sampler_steps_list:
        fake_feat_batches: list[torch.Tensor] = []
        out_collected: list[torch.Tensor] = []
        lf_collected: list[torch.Tensor] = []
        zh_collected: list[torch.Tensor] = []
        n_collected = 0
        for z_h_cpu, shape in zip(z_h_store, shapes):
            z_h_gpu = z_h_cpu.to(device)
            z_l_fake = sample_macro_latents(model, shape=shape, method=sampler, num_steps=steps, device=str(device))
            z_fake = reconstruct_from_decomposition(z_l_fake, z_h_gpu)
            z_l_up = reconstruct_low_freq(z_l_fake)
            imgs = decode_latents(vae, z_fake)
            fake_feat_batches.append(fid_model(imgs))
            if n_collected < num_sample_images:
                out_collected.append(imgs.cpu())
                lf_collected.append(decode_latents(vae, z_l_up).cpu())
                zh_collected.append(decode_latents(vae, z_h_gpu).cpu())
                n_collected += imgs.shape[0]
        fid_results[steps] = calculate_fid(real_stats, stats_from_feature_batches(fake_feat_batches))
        if out_collected:
            n = num_sample_images
            sample_components[steps] = {
                "output": torch.cat(out_collected, dim=0)[:n],
                "low_freq": torch.cat(lf_collected, dim=0)[:n],
                "z_h": torch.cat(zh_collected, dim=0)[:n],
            }

    if was_training:
        model.train()
    return fid_results, sample_components


def print_model_summary(model: torch.nn.Module, dummy_inputs: tuple) -> None:
    """Print per-layer input/output shapes by running one dry forward pass with hooks."""

    def _fmt_shape(x: object) -> str:
        if isinstance(x, torch.Tensor):
            return "[" + ", ".join(str(d) for d in x.shape) + "]"
        if isinstance(x, (tuple, list)):
            parts = [_fmt_shape(t) for t in x if isinstance(t, torch.Tensor)]
            return " | ".join(parts) if parts else "—"
        return "—"

    records: list[tuple[str, str, str, str, int]] = []
    handles = []

    for name, module in model.named_modules():
        def _make_hook(n: str):
            def _hook(mod: torch.nn.Module, inp: tuple, out: object) -> None:
                n_params = sum(p.numel() for p in mod.parameters(recurse=False))
                records.append((n or "(root)", type(mod).__name__, _fmt_shape(inp), _fmt_shape(out), n_params))
            return _hook
        handles.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        try:
            model(*dummy_inputs)
        except Exception as exc:
            print(f"[summary] dry forward pass failed: {exc}")
    model.train()

    for h in handles:
        h.remove()

    col = (4, 38, 26, 26, 26, 11)
    total = sum(col) + len(col) + 1
    header = (
        f"{'#':<{col[0]}} {'Name':<{col[1]}} {'Type':<{col[2]}} "
        f"{'Input':<{col[3]}} {'Output':<{col[4]}} {'Params':>{col[5]}}"
    )
    sep = "─" * total

    print()
    print("Model Summary")
    print("═" * total)
    print(header)
    print(sep)
    for i, (name, typ, in_s, out_s, n) in enumerate(records):
        depth = name.count(".")
        indent = "  " * depth
        short = (indent + name.rsplit(".", 1)[-1]) if "." in name else (indent + name)
        short = short[: col[1] - 1]
        print(
            f"{i:<{col[0]}} {short:<{col[1]}} {typ:<{col[2]}} "
            f"{in_s:<{col[3]}} {out_s:<{col[4]}} {n:>{col[5]},}"
        )
    print("═" * total)
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_p:,}   Trainable: {trainable_p:,}")
    print()


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

    output_cfg = config.get("output", {})
    out_dir = Path(args.out if args.out != "outputs/macro_flow" else output_cfg.get("dir", args.out))
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

    if accelerator.is_main_process:
        input_shape = [int(d) for d in model_cfg.get("input_shape", [4, 16, 16])]
        _B = 2
        _dummy_x = torch.randn(_B, *input_shape)
        _dummy_t = torch.rand(_B)
        _dummy_dt = torch.zeros(_B)
        print_model_summary(model, (_dummy_x, _dummy_t, _dummy_dt))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", training.get("lr", 2.0e-4))),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    ema = EMA(accelerator.unwrap_model(model), float(training.get("ema_decay", 0.9999))) if training.get("ema", True) else None
    start_step = 0

    resume_path = args.resume or output_cfg.get("resume") or None
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
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
    sample_every = int(logging_cfg.get("sample_every_steps", 0))
    fid_num_samples = args.fid_num_samples or int(evaluation_cfg.get("fid_num_samples", 512))
    fid_batch_size = int(evaluation_cfg.get("fid_batch_size", min(batch_size, 32)))
    fid_sampler = str(evaluation_cfg.get("sampler", "heun"))
    raw_steps = evaluation_cfg.get("sampler_steps", 50)
    fid_sampler_steps: list[int] = raw_steps if isinstance(raw_steps, list) else [int(raw_steps)]
    num_sample_images = int(evaluation_cfg.get("num_sample_images", 16))

    vae = None
    fid_model = None
    fid_dataset = None
    precomputed_fid_stats = None
    sample_z_h: torch.Tensor | None = None
    if accelerator.is_main_process and (fid_every > 0 or sample_every > 0):
        vae_cfg = config.get("vae", {})
        vae = load_sd_vae(
            checkpoint=resolve_vae_checkpoint(vae_cfg),
            cache_dir=str(vae_cfg.get("cache_dir", ".cache/huggingface")),
            dtype=str(vae_cfg.get("dtype", "fp16")),
            device=str(accelerator.device),
        )
        fid_dataset = CachedMicroLatentDataset(latent_dir)
        # Pre-fetch a fixed small batch of z_H for lightweight sampling visualization.
        _tmp = DataLoader(fid_dataset, batch_size=num_sample_images, shuffle=False, num_workers=0)
        sample_z_h = next(iter(_tmp))["z_H"][:num_sample_images].to(accelerator.device)
        del _tmp
        if fid_every > 0:
            fid_model = InceptionFID(accelerator.device)
            fid_stats_path = evaluation_cfg.get("fid_stats")
            if fid_stats_path:
                p = Path(fid_stats_path)
                if p.exists():
                    precomputed_fid_stats = load_precomputed_fid_stats(p)
                    accelerator.print(f"loaded pre-computed FID stats from {p}")
                else:
                    accelerator.print(f"WARNING: fid_stats path {fid_stats_path!r} not found; computing real stats from dataset")
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

            if sample_every > 0 and step % sample_every == 0 and accelerator.is_main_process and tracker == "wandb" and sample_z_h is not None:
                import wandb
                eval_model = accelerator.unwrap_model(model)
                eval_model.eval()
                with torch.no_grad():
                    n = sample_z_h.shape[0]
                    z_l_fake = sample_macro_latents(
                        eval_model,
                        shape=(n, eval_model.in_channels, eval_model.resolution, eval_model.resolution),
                        method=fid_sampler,
                        num_steps=fid_sampler_steps[0],
                        device=str(accelerator.device),
                    )
                    imgs_output = decode_latents(vae, reconstruct_from_decomposition(z_l_fake, sample_z_h)).cpu()
                    imgs_low_freq = decode_latents(vae, reconstruct_low_freq(z_l_fake)).cpu()
                    imgs_zh = decode_latents(vae, sample_z_h).cpu()
                def _to_wandb(tensors: torch.Tensor) -> list:
                    return [
                        wandb.Image(img.float().clamp(-1, 1).add(1).div(2).permute(1, 2, 0).numpy())
                        for img in tensors
                    ]
                s = fid_sampler_steps[0]
                wandb.log(
                    {
                        f"samples/output_{s}step": _to_wandb(imgs_output),
                        f"samples/low_freq_{s}step": _to_wandb(imgs_low_freq),
                        f"samples/z_h_{s}step": _to_wandb(imgs_zh),
                    },
                    step=step,
                )
                eval_model.train()

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
                    fid_results, sample_components = evaluate_multi_step_fid(
                        eval_model,
                        fid_dataset,
                        vae,
                        fid_model,
                        num_samples=min(fid_num_samples, len(fid_dataset)),
                        batch_size=fid_batch_size,
                        sampler=fid_sampler,
                        sampler_steps_list=fid_sampler_steps,
                        device=accelerator.device,
                        real_stats=precomputed_fid_stats,
                        num_sample_images=num_sample_images,
                    )
                    if backup is not None:
                        with torch.no_grad():
                            for name, parameter in eval_model.named_parameters():
                                if name in backup:
                                    parameter.copy_(backup[name])
                    fid_logs = {f"eval/fid_{s}step": v for s, v in fid_results.items()}
                    accelerator.log(fid_logs, step=step)
                    for s, v in fid_results.items():
                        accelerator.print(f"step={step} fid_{s}step={v:.4f}")
                    if tracker == "wandb" and sample_components:
                        import wandb
                        wandb_imgs: dict[str, list] = {}
                        for s, components in sample_components.items():
                            for key, tensors in components.items():
                                wandb_imgs[f"eval/{key}_{s}step"] = [
                                    wandb.Image(img.float().clamp(-1, 1).add(1).div(2).permute(1, 2, 0).numpy())
                                    for img in tensors
                                ]
                        wandb.log(wandb_imgs, step=step)
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
