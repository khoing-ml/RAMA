from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_dtype = "fp16" if default_device == "cuda" else "fp32"
    parser = argparse.ArgumentParser(description="Run the full discrete-token Latent-RAMA training pipeline.")

    parser.add_argument("--images", default="data/celeba256", help="Image folder or Parquet shard root.")
    parser.add_argument("--latents", default="data/latents", help="Cached SD-VAE latent directory.")
    parser.add_argument("--vae-checkpoint", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--hf-cache-dir", default=".cache/huggingface")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--dtype", default=default_dtype, choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--device", default=default_device)

    parser.add_argument("--macro-config", default="configs/celeba256_sdvae_macro.yaml")
    parser.add_argument("--micro-config", default="configs/celeba256_sdvae_micro.yaml")
    parser.add_argument("--macro-out", default="outputs/macro_flow")
    parser.add_argument("--micro-out", default="outputs/micro_rama")
    parser.add_argument("--bases", default="cache/rama_bases_p256_d16.pt")
    parser.add_argument("--tokenizer-config", default="cache/rama_tokenizer_config.pt")

    parser.add_argument("--cache-batch-size", type=int, default=32)
    parser.add_argument("--cache-num-workers", type=int, default=4)
    parser.add_argument("--macro-batch-size", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--train-num-workers", type=int, default=4)
    parser.add_argument("--macro-max-steps", type=int, default=None)
    parser.add_argument("--micro-max-steps", type=int, default=None)
    parser.add_argument("--quant-batch-size", type=int, default=16)
    parser.add_argument("--quant-max-batches", type=int, default=200)
    parser.add_argument("--quant-percentile", type=float, default=99.5)
    parser.add_argument("--num-bins", type=int, default=256)

    parser.add_argument("--sample-num-samples", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sampler", choices=("heun", "euler"), default="heun")
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-argmax", action="store_true", help="Use argmax micro tokens for final sampling.")

    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--force-cache", action="store_true", help="Cache latents even if --latents already has .pt files.")
    parser.add_argument("--skip-macro", action="store_true")
    parser.add_argument("--skip-quant-reconstruction", action="store_true")
    parser.add_argument("--skip-micro", action="store_true")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true", default=True)
    parser.add_argument("--enable-wandb", action="store_false", dest="disable_wandb")
    parser.add_argument("--use-accelerate", action="store_true", help="Launch train scripts through accelerate.")
    parser.add_argument("--accelerate-num-processes", type=int, default=None)
    parser.add_argument("--accelerate-mixed-precision", default=None, choices=("no", "fp16", "bf16"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def config_total_steps(config_path: str | Path) -> int:
    config = load_yaml(config_path)
    training = config.get("training", {})
    return int(training.get("total_steps", training.get("max_steps", 0)))


def checkpoint_path(out_dir: str | Path, step: int | None) -> Path | None:
    if step is None:
        return None
    return Path(out_dir) / "checkpoints" / f"step_{step:08d}.pt"


def latest_checkpoint(out_dir: str | Path) -> Path:
    checkpoints = sorted((Path(out_dir) / "checkpoints").glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found under {Path(out_dir) / 'checkpoints'}")
    return checkpoints[-1]


def expected_or_latest_checkpoint(out_dir: str | Path, step: int | None) -> Path:
    expected = checkpoint_path(out_dir, step)
    if expected is not None and expected.exists():
        return expected
    return latest_checkpoint(out_dir)


def planned_checkpoint(out_dir: str | Path, step: int | None) -> Path:
    if step is not None and step > 0:
        return Path(out_dir) / "checkpoints" / f"step_{step:08d}.pt"
    return Path(out_dir) / "checkpoints" / "step_<latest>.pt"


def has_latents(path: str | Path) -> bool:
    root = Path(path)
    return root.exists() and any(root.rglob("*.pt"))


def add_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def train_command(
    script: str,
    use_accelerate: bool,
    num_processes: int | None = None,
    mixed_precision: str | None = None,
) -> list[str]:
    if use_accelerate:
        command = ["accelerate", "launch"]
        if num_processes is not None:
            if num_processes > 1:
                command.append("--multi_gpu")
            command.extend(["--num_processes", str(num_processes)])
        if mixed_precision is not None:
            command.extend(["--mixed_precision", mixed_precision])
        command.append(script)
        return command
    return [sys.executable, script]


def run(command: list[str], dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    macro_steps = args.macro_max_steps or config_total_steps(args.macro_config)
    micro_steps = args.micro_max_steps or config_total_steps(args.micro_config)

    if not args.skip_cache and (args.force_cache or not has_latents(args.latents)):
        command = [
            sys.executable,
            "scripts/cache_sdvae_latents.py",
            "--images",
            args.images,
            "--out",
            args.latents,
            "--checkpoint",
            args.vae_checkpoint,
            "--cache-dir",
            args.hf_cache_dir,
            "--batch-size",
            str(args.cache_batch_size),
            "--num-workers",
            str(args.cache_num_workers),
            "--image-size",
            str(args.image_size),
            "--dtype",
            args.dtype,
            "--device",
            args.device,
            "--store-components",
        ]
        run(command, args.dry_run)
    else:
        print(f"Skipping latent cache; found cached latents under {args.latents}")

    if not args.skip_macro:
        command = train_command(
            "scripts/train_macro_flow.py",
            args.use_accelerate,
            args.accelerate_num_processes,
            args.accelerate_mixed_precision,
        )
        command.extend(["--config", args.macro_config, "--latents", args.latents, "--out", args.macro_out])
        command.extend(["--num-workers", str(args.train_num_workers)])
        add_optional(command, "--batch-size", args.macro_batch_size)
        add_optional(command, "--max-steps", args.macro_max_steps)
        if args.disable_wandb:
            command.append("--disable-wandb")
        run(command, args.dry_run)
    else:
        print("Skipping macro training")

    command = [
        sys.executable,
        "scripts/estimate_quant_bound.py",
        "--latent-cache",
        args.latents,
        "--bases",
        args.bases,
        "--output",
        args.tokenizer_config,
        "--num-bins",
        str(args.num_bins),
        "--percentile",
        str(args.quant_percentile),
        "--max-batches",
        str(args.quant_max_batches),
        "--batch-size",
        str(args.quant_batch_size),
        "--device",
        args.device,
    ]
    run(command, args.dry_run)

    if not args.skip_quant_reconstruction:
        command = [
            sys.executable,
            "scripts/test_quant_reconstruction.py",
            "--latent-cache",
            args.latents,
            "--bases",
            args.bases,
            "--tokenizer-config",
            args.tokenizer_config,
            "--out",
            "outputs/quantization_tests/vae_vs_macro_vs_quant.png",
            "--checkpoint",
            args.vae_checkpoint,
            "--cache-dir",
            args.hf_cache_dir,
            "--dtype",
            args.dtype,
            "--device",
            args.device,
        ]
        run(command, args.dry_run)
    else:
        print("Skipping quantization reconstruction")

    if not args.skip_micro:
        command = train_command(
            "scripts/train_micro_rama.py",
            args.use_accelerate,
            args.accelerate_num_processes,
            args.accelerate_mixed_precision,
        )
        command.extend(
            [
                "--config",
                args.micro_config,
                "--latents",
                args.latents,
                "--out",
                args.micro_out,
                "--micro-type",
                "categorical",
                "--tokenizer-config",
                args.tokenizer_config,
                "--bases",
                args.bases,
                "--num-workers",
                str(args.train_num_workers),
            ]
        )
        add_optional(command, "--batch-size", args.micro_batch_size)
        add_optional(command, "--max-steps", args.micro_max_steps)
        if args.disable_wandb:
            command.append("--disable-wandb")
        run(command, args.dry_run)
    else:
        print("Skipping micro training")

    if args.skip_sampling:
        print("Skipping sampling")
        return

    if args.dry_run:
        macro_checkpoint = planned_checkpoint(args.macro_out, macro_steps if macro_steps > 0 else None)
        micro_checkpoint = planned_checkpoint(args.micro_out, micro_steps if micro_steps > 0 else None)
    else:
        macro_checkpoint = expected_or_latest_checkpoint(args.macro_out, macro_steps if macro_steps > 0 else None)
        micro_checkpoint = expected_or_latest_checkpoint(args.micro_out, micro_steps if micro_steps > 0 else None)

    command = [
        sys.executable,
        "scripts/sample_macro_flow.py",
        "--checkpoint",
        str(macro_checkpoint),
        "--out",
        "outputs/macro_samples/macro_only_samples.png",
        "--num-samples",
        str(args.sample_num_samples),
        "--sampler",
        args.sampler,
        "--steps",
        str(args.sample_steps),
        "--vae-checkpoint",
        args.vae_checkpoint,
        "--cache-dir",
        args.hf_cache_dir,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
    ]
    run(command, args.dry_run)

    command = [
        sys.executable,
        "scripts/reconstruct_micro_real_zl.py",
        "--micro-checkpoint",
        str(micro_checkpoint),
        "--latent-cache",
        args.latents,
        "--bases",
        args.bases,
        "--tokenizer-config",
        args.tokenizer_config,
        "--out",
        "outputs/micro_reconstructions/real_zL_macro_vs_micro_argmax.png",
        "--vae-checkpoint",
        args.vae_checkpoint,
        "--cache-dir",
        args.hf_cache_dir,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
    ]
    run(command, args.dry_run)

    command = [
        sys.executable,
        "scripts/sample_full_model.py",
        "--macro-checkpoint",
        str(macro_checkpoint),
        "--micro-checkpoint",
        str(micro_checkpoint),
        "--bases",
        args.bases,
        "--tokenizer-config",
        args.tokenizer_config,
        "--out",
        "outputs/full_samples/generated_zL_macro_plus_micro.png",
        "--macro-out",
        "outputs/full_samples/generated_zL_macro_only.png",
        "--num-samples",
        str(args.sample_num_samples),
        "--sampler",
        args.sampler,
        "--steps",
        str(args.sample_steps),
        "--temperature",
        str(args.sample_temperature),
        "--vae-checkpoint",
        args.vae_checkpoint,
        "--cache-dir",
        args.hf_cache_dir,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
    ]
    if args.sample_argmax:
        command.append("--use-argmax")
    run(command, args.dry_run)

    print("\nEnd-to-end discrete RAMA pipeline finished.")
    print(f"Macro checkpoint: {macro_checkpoint}")
    print(f"Micro checkpoint: {micro_checkpoint}")


if __name__ == "__main__":
    main()
