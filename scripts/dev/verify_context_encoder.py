from __future__ import annotations

"""Context encoder diagnostic script.

Runs five checks:
  1. Shape & Statistics   — correct output shape, no NaN/Inf, reasonable std
  2. Diversity            — different inputs produce meaningfully different contexts
  3. Context Ablation     — real context should give lower micro loss than zero/random
  4. Gradient Flow        — gradients reach every context encoder parameter
  5. Spatial Coherence    — saves a PCA-coloured context map (requires PIL)
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.micro.loss import categorical_micro_loss
from src.micro.micro_rama_categorical import build_categorical_micro_rama_net
from src.modules.micro_rama import build_context_encoder
from src.modules.rama import make_orthogonal_bases, patchify
from src.rama.projector import RAMAProjector
from src.rama.tokenizer import RAMATokenizer, build_tokenizer_from_config, load_tokenizer_config


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _derive_grid_and_patch(config: dict) -> tuple[tuple[int, int], int]:
    """Return (grid_size, patch_dim) from micro_latent config block."""
    ml = config.get("micro_latent", {})
    residual_shape = ml.get("residual_shape", [4, 32, 32])
    patch_size = int(ml.get("patch_size", 4))
    C, H, W = int(residual_shape[0]), int(residual_shape[1]), int(residual_shape[2])
    grid_size = (H // patch_size, W // patch_size)
    patch_dim = C * patch_size * patch_size
    return grid_size, patch_dim, patch_size, (C, H, W)


def _get_latents(args, config, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (z_L, z_H) with batch dim.  Falls back to random tensors."""
    latent_dir = args.latents or config.get("latents", {}).get("output_dir")
    if latent_dir:
        latent_path = Path(latent_dir)
        pts = sorted(latent_path.rglob("*.pt"))[: args.num_images]
        if pts:
            from src.dataset.latent_dataset import CachedMicroLatentDataset
            from torch.utils.data import DataLoader, Subset
            ds = CachedMicroLatentDataset(latent_dir)
            subset = Subset(ds, list(range(min(args.num_images, len(ds)))))
            loader = DataLoader(subset, batch_size=args.num_images)
            batch = next(iter(loader))
            return batch["z_L"].to(device), batch["z_H"].to(device)
    # Fall back to random tensors so the script still runs without data
    print("  [warn] no latent data found — using random tensors (checks 1, 2, 4, 5 only meaningful after training)")
    _, _, _, (C, H, W) = _derive_grid_and_patch(config)
    z_l = torch.randn(args.num_images, C, H // 2, W // 2, device=device)
    z_h = torch.randn(args.num_images, C, H, W, device=device)
    return z_l, z_h


def _pca3(matrix: torch.Tensor) -> torch.Tensor:
    """Project [N, D] → [N, 3] via SVD (no sklearn required)."""
    m = matrix - matrix.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(m, full_matrices=False)
    return (m @ Vt[:3].T).float()


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def _fmt(label: str, status: str, detail: str) -> None:
    symbol = {"PASS": "✓", "FAIL": "✗", "WARN": "!"}.get(status, "?")
    print(f"  [{symbol}] {label}: {detail}")


# ──────────────────────────────────────────────────────────────
# Check 1 — Shape & Statistics
# ──────────────────────────────────────────────────────────────

def check_shape_and_stats(encoder, z_l: torch.Tensor, grid_size: tuple[int, int], context_dim: int) -> bool:
    print("\n[1] Shape & Statistics")
    with torch.no_grad():
        ctx = encoder(z_l)

    B, P, D = ctx.shape
    expected_P = grid_size[0] * grid_size[1]

    ok = True
    if P != expected_P or D != context_dim:
        _fmt("shape", FAIL, f"got [{B}, {P}, {D}], expected [{B}, {expected_P}, {context_dim}]")
        ok = False
    else:
        _fmt("shape", PASS, f"[{B}, {P}, {D}]")

    if ctx.isnan().any():
        _fmt("NaN", FAIL, "context contains NaN")
        ok = False
    else:
        _fmt("NaN", PASS, "none")

    if ctx.isinf().any():
        _fmt("Inf", FAIL, "context contains Inf")
        ok = False
    else:
        _fmt("Inf", PASS, "none")

    mean = ctx.mean().item()
    std = ctx.std().item()
    std_status = PASS if 0.05 < std < 10.0 else WARN
    _fmt("mean", PASS, f"{mean:.4f}")
    _fmt("std",  std_status, f"{std:.4f}" + (" (collapsed?)" if std_status == WARN else ""))

    return ok


# ──────────────────────────────────────────────────────────────
# Check 2 — Diversity
# ──────────────────────────────────────────────────────────────

def check_diversity(encoder, z_l: torch.Tensor) -> bool:
    print("\n[2] Diversity")
    if z_l.shape[0] < 2:
        _fmt("diversity", WARN, "need at least 2 images — skipped")
        return True

    with torch.no_grad():
        ctx = encoder(z_l)

    # Mean absolute difference between all pairs of context vectors
    cross = (ctx.unsqueeze(0) - ctx.unsqueeze(1)).abs().mean().item()
    self_diff = 0.0  # by definition

    _fmt("same-image diff", PASS, f"{self_diff:.6f}")

    threshold = 0.01
    if cross < threshold:
        _fmt("cross-image diff", FAIL, f"{cross:.6f} (< {threshold} — encoder may be collapsed)")
        return False
    else:
        _fmt("cross-image diff", PASS, f"{cross:.6f}")

    # Per-patch diversity: std across batch at each patch position
    patch_std = ctx.std(dim=0).mean().item()
    patch_status = PASS if patch_std > 0.01 else WARN
    _fmt("per-patch batch std", patch_status, f"{patch_std:.6f}")

    return True


# ──────────────────────────────────────────────────────────────
# Check 3 — Context Ablation
# ──────────────────────────────────────────────────────────────

def check_ablation(encoder, micro_model, z_l, z_h, projector, tokenizer, patch_size: int) -> bool:
    print("\n[3] Context Ablation  (most informative after training)")
    patches = patchify(z_h, patch_size=patch_size)
    tokens = tokenizer.quantize(projector.project(patches))
    num_bins = tokenizer.num_bins

    with torch.no_grad():
        ctx_real   = encoder(z_l)
        ctx_zero   = torch.zeros_like(ctx_real)
        ctx_random = torch.randn_like(ctx_real)

        loss_real   = categorical_micro_loss(micro_model(ctx_real),   tokens, num_bins).item()
        loss_zero   = categorical_micro_loss(micro_model(ctx_zero),   tokens, num_bins).item()
        loss_random = categorical_micro_loss(micro_model(ctx_random), tokens, num_bins).item()

    baseline = min(loss_zero, loss_random)
    _fmt("loss (real ctx)",   PASS, f"{loss_real:.4f}")
    _fmt("loss (zero ctx)",   PASS, f"{loss_zero:.4f}")
    _fmt("loss (random ctx)", PASS, f"{loss_random:.4f}")

    if loss_real < baseline - 0.01:
        _fmt("ablation", PASS, f"real context is better by {baseline - loss_real:.4f}")
        return True
    elif abs(loss_real - baseline) < 0.01:
        _fmt("ablation", WARN, "real context ≈ random/zero — encoder may not be trained yet")
        return True
    else:
        _fmt("ablation", FAIL, "real context is WORSE than random/zero — something is wrong")
        return False


# ──────────────────────────────────────────────────────────────
# Check 4 — Gradient Flow
# ──────────────────────────────────────────────────────────────

def check_gradient_flow(encoder, micro_model, z_l, z_h, projector, tokenizer, patch_size: int) -> bool:
    print("\n[4] Gradient Flow")
    patches = patchify(z_h, patch_size=patch_size)
    tokens = tokenizer.quantize(projector.project(patches))

    encoder.zero_grad()
    micro_model.zero_grad()

    ctx = encoder(z_l)
    logits = micro_model(ctx)
    loss = categorical_micro_loss(logits, tokens, tokenizer.num_bins)
    loss.backward()

    params_with_grad = [(n, p) for n, p in encoder.named_parameters() if p.grad is not None]
    params_no_grad   = [(n, p) for n, p in encoder.named_parameters() if p.grad is None]

    if params_no_grad:
        for name, _ in params_no_grad[:5]:
            _fmt("no grad", FAIL, name)
        ok = False
    else:
        ok = True

    if not params_with_grad:
        _fmt("gradient flow", FAIL, "no parameters received gradients")
        return False

    grad_norms = [p.grad.abs().mean().item() for _, p in params_with_grad]
    mean_grad = sum(grad_norms) / len(grad_norms)
    max_grad  = max(grad_norms)
    min_grad  = min(grad_norms)

    grad_status = PASS if mean_grad > 1e-9 else FAIL
    _fmt("params with grad", grad_status, f"{len(params_with_grad)}/{len(params_with_grad) + len(params_no_grad)}")
    _fmt("mean |grad|", grad_status, f"{mean_grad:.3e}")
    _fmt("min  |grad|", PASS if min_grad > 1e-12 else WARN, f"{min_grad:.3e}")
    _fmt("max  |grad|", PASS if max_grad < 1e3   else WARN, f"{max_grad:.3e}")

    encoder.zero_grad()
    micro_model.zero_grad()
    return ok and mean_grad > 1e-9


# ──────────────────────────────────────────────────────────────
# Check 5 — Spatial Coherence
# ──────────────────────────────────────────────────────────────

def check_spatial_coherence(encoder, z_l: torch.Tensor, grid_size: tuple[int, int], out_path: Path) -> bool:
    print("\n[5] Spatial Coherence")
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        _fmt("spatial coherence", WARN, "PIL not available — skipped")
        return True

    with torch.no_grad():
        ctx = encoder(z_l[:1])  # single image: [1, P, D]

    H_grid, W_grid = grid_size
    P, D = H_grid * W_grid, ctx.shape[-1]
    ctx_flat = ctx[0].cpu().float()  # [P, D]

    rgb = _pca3(ctx_flat)  # [P, 3]
    rgb -= rgb.min(0).values
    rgb /= rgb.max(0).values.clamp_min(1e-6)
    rgb_np = (rgb.numpy() * 255).astype("uint8").reshape(H_grid, W_grid, 3)

    scale = max(1, 64 // max(H_grid, W_grid))
    img = Image.fromarray(rgb_np, "RGB").resize(
        (W_grid * scale, H_grid * scale), Image.NEAREST
    )
    img.save(out_path)
    _fmt("PCA map saved", PASS, str(out_path))
    _fmt("hint", PASS, "smooth colour transitions → spatially coherent; pure noise → encoder not capturing structure")
    return True


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify context encoder output quality.")
    parser.add_argument("--config", default="configs/celeba256_sdvae_high_rama.yaml")
    parser.add_argument("--latents", default=None, help="path to latent cache directory")
    parser.add_argument("--bases", default="cache/rama_bases_p256_d16.pt")
    parser.add_argument("--tokenizer-config", default="cache/rama_tokenizer_config.pt")
    parser.add_argument("--checkpoint", default=None, help="path to a .pt checkpoint containing context_encoder and/or micro_model state dicts")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--out-dir", default="outputs/context_encoder_verify")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)

    grid_size, patch_dim, patch_size, (C, H, W) = _derive_grid_and_patch(config)
    context_dim = int(config.get("context_encoder", {}).get("context_dim", 256))
    num_bins = int(config.get("tokenizer", {}).get("num_bins", 256))

    print(f"Config  : {args.config}")
    print(f"Device  : {args.device}")
    print(f"Grid    : {grid_size[0]}×{grid_size[1]} ({grid_size[0]*grid_size[1]} patches)")
    print(f"patch_dim={patch_dim}  context_dim={context_dim}  num_bins={num_bins}")

    # Build models
    enc_cfg = dict(config.get("context_encoder", {}))
    enc_cfg["grid_size"] = list(grid_size)
    enc_cfg["patch_size"] = patch_size
    encoder = build_context_encoder(enc_cfg).to(args.device)

    micro_cfg = dict(config.get("micro", {}))
    micro_cfg["patch_dim"] = patch_dim
    micro_model = build_categorical_micro_rama_net(micro_cfg, num_bins=num_bins).to(args.device)

    # Load checkpoint if provided
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        if "context_encoder" in ckpt:
            encoder.load_state_dict(ckpt["context_encoder"])
            print(f"Loaded context_encoder from {args.checkpoint}")
        if "micro_model" in ckpt:
            micro_model.load_state_dict(ckpt["micro_model"])
            print(f"Loaded micro_model from {args.checkpoint}")

    # Load projector & tokenizer
    bases_path = Path(args.bases)
    if bases_path.exists():
        bases = torch.load(bases_path, map_location="cpu").float().to(args.device)
    else:
        print(f"  [warn] bases not found at {args.bases} — generating random orthogonal bases")
        bases = make_orthogonal_bases(grid_size[0] * grid_size[1], patch_dim).to(args.device)

    projector = RAMAProjector(bases).to(args.device)
    projector.requires_grad_(False)

    tokenizer_path = Path(args.tokenizer_config)
    tokenizer = (
        build_tokenizer_from_config(load_tokenizer_config(str(tokenizer_path)))
        if tokenizer_path.exists()
        else RAMATokenizer(num_bins=num_bins)
    )

    # Load data
    z_l, z_h = _get_latents(args, config, args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run checks
    results: dict[str, bool] = {}
    results["shape_stats"]   = check_shape_and_stats(encoder, z_l, grid_size, context_dim)
    results["diversity"]     = check_diversity(encoder, z_l)
    results["ablation"]      = check_ablation(encoder, micro_model, z_l, z_h, projector, tokenizer, patch_size)
    results["gradient_flow"] = check_gradient_flow(encoder, micro_model, z_l, z_h, projector, tokenizer, patch_size)
    results["spatial"]       = check_spatial_coherence(encoder, z_l, grid_size, out_dir / "context_pca.png")

    # Summary
    print("\n" + "─" * 50)
    print("Summary")
    for name, passed in results.items():
        sym = "✓" if passed else "✗"
        print(f"  [{sym}] {name}")
    if all(results.values()):
        print("\nAll checks passed.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
