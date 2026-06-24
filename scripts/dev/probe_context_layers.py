"""Probe each DiT context encoder layer to visualise representation quality.

Hooks into every DiTContextBlock to capture intermediate token representations,
then produces a multi-panel figure:

  Row 1 — PCA spatial maps after each layer (smooth colours = spatial structure)
  Row 2 — Per-token L2 norm distribution per layer (boxplot)
  Row 3 — Layer-to-layer cosine similarity (how much each block changes tokens)
  Row 4 — Fraction of variance explained by top-3 PCs per layer

Run:
  .venv/bin/python scripts/dev/probe_context_layers.py
  .venv/bin/python scripts/dev/probe_context_layers.py --checkpoint path/to/ckpt.pt
  .venv/bin/python scripts/dev/probe_context_layers.py --config configs/debug_6gb_micro.yaml --latents data/debug_micro_latents
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.micro_rama import build_context_encoder, DiTContextBlock


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def derive_grid_and_patch(config: dict):
    ml = config.get("micro_latent", {})
    residual_shape = ml.get("residual_shape", [4, 32, 32])
    patch_size = int(ml.get("patch_size", 4))
    C, H, W = int(residual_shape[0]), int(residual_shape[1]), int(residual_shape[2])
    grid_size = (H // patch_size, W // patch_size)
    patch_dim = C * patch_size * patch_size
    return grid_size, patch_dim, patch_size, (C, H, W)


def get_latents(args, config, device) -> torch.Tensor:
    latent_dir = args.latents or config.get("latents", {}).get("output_dir")
    if latent_dir:
        latent_path = Path(latent_dir)
        pts = sorted(latent_path.rglob("*.pt"))[: args.num_images]
        if pts:
            try:
                from src.dataset.latent_dataset import CachedMicroLatentDataset
                from torch.utils.data import DataLoader, Subset
                ds = CachedMicroLatentDataset(latent_dir)
                subset = Subset(ds, list(range(min(args.num_images, len(ds)))))
                loader = DataLoader(subset, batch_size=args.num_images)
                batch = next(iter(loader))
                return batch["z_L"].to(device)
            except Exception as e:
                print(f"  [warn] could not load latents: {e}")
    _, _, _, (C, H, W) = derive_grid_and_patch(config)
    print("  [warn] using random z_L tensors — PCA maps will be meaningless until a checkpoint is provided")
    return torch.randn(args.num_images, C, H // 2, W // 2, device=device)


def pca3(matrix: torch.Tensor) -> torch.Tensor:
    """[N, D] → [N, 3] via SVD, no sklearn needed."""
    m = matrix.float() - matrix.float().mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(m, full_matrices=False)
    return m @ Vt[:3].T


def variance_explained(matrix: torch.Tensor, k: int = 3) -> float:
    """Fraction of variance captured by top-k PCs."""
    m = matrix.float() - matrix.float().mean(0, keepdim=True)
    S = torch.linalg.svdvals(m)
    var = S.pow(2)
    return (var[:k].sum() / var.sum().clamp_min(1e-9)).item()


def cosine_sim_layers(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between corresponding tokens in two layers.
    a, b: [P, D]"""
    a = a.float()
    b = b.float()
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return (a * b).sum(-1).mean().item()


# ──────────────────────────────────────────────────────────────
# Hook collection
# ──────────────────────────────────────────────────────────────

def collect_layer_activations(encoder, z_l: torch.Tensor) -> tuple[list[torch.Tensor], list[str]]:
    """Return per-layer token tensors [P, D] (averaged over batch) and labels."""
    acts: list[torch.Tensor] = []
    labels: list[str] = []
    hooks = []

    # After the input projection
    def make_hook(label):
        def hook(module, input, output):
            acts.append(output.detach().cpu().mean(0))  # [P, D]
            labels.append(label)
        return hook

    hooks.append(encoder.input_proj.register_forward_hook(make_hook("proj")))

    if hasattr(encoder, "blocks"):
        for i, block in enumerate(encoder.blocks):
            hooks.append(block.register_forward_hook(make_hook(f"block {i}")))
    elif hasattr(encoder, "transformer_blocks"):
        for i, block in enumerate(encoder.transformer_blocks):
            hooks.append(block.register_forward_hook(make_hook(f"block {i}")))

    with torch.no_grad():
        encoder(z_l)

    for h in hooks:
        h.remove()

    # input_proj outputs [B, P, D]; DiTContextBlock outputs [B, P, D] — both already [P,D] after .mean(0)
    return acts, labels


def collect_per_token_norms(encoder, z_l: torch.Tensor) -> tuple[list[torch.Tensor], list[str]]:
    """Return per-layer [B*P] norm tensors and labels."""
    acts_norm: list[torch.Tensor] = []
    labels: list[str] = []
    hooks = []

    def make_hook(label):
        def hook(module, input, output):
            norms = output.detach().cpu().float().norm(dim=-1).reshape(-1)  # [B*P]
            acts_norm.append(norms)
            labels.append(label)
        return hook

    hooks.append(encoder.input_proj.register_forward_hook(make_hook("proj")))

    if hasattr(encoder, "blocks"):
        for i, block in enumerate(encoder.blocks):
            hooks.append(block.register_forward_hook(make_hook(f"block {i}")))
    elif hasattr(encoder, "transformer_blocks"):
        for i, block in enumerate(encoder.transformer_blocks):
            hooks.append(block.register_forward_hook(make_hook(f"block {i}")))

    with torch.no_grad():
        encoder(z_l)

    for h in hooks:
        h.remove()

    return acts_norm, labels


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def make_figure(
    acts: list[torch.Tensor],
    norms: list[torch.Tensor],
    labels: list[str],
    grid_size: tuple[int, int],
    out_path: Path,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [warn] matplotlib not available — install it to produce figures")
        return

    n = len(acts)
    H_grid, W_grid = grid_size

    fig = plt.figure(figsize=(max(n * 2.5, 12), 14))
    fig.suptitle("Context Encoder Layer Probing", fontsize=14, y=0.98)

    # ── Row 1: PCA spatial maps ──────────────────────────────
    for i, (act, label) in enumerate(zip(acts, labels)):
        ax = fig.add_subplot(4, n, i + 1)
        rgb = pca3(act)                          # [P, 3]
        rgb -= rgb.min(0).values
        rgb /= rgb.max(0).values.clamp_min(1e-6)
        rgb_np = rgb.numpy().reshape(H_grid, W_grid, 3)
        ax.imshow(rgb_np)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("PCA map", fontsize=8)

    # ── Row 2: per-token norm boxplot ────────────────────────
    ax_norm = fig.add_subplot(4, 1, 2)
    data = [n_.numpy() for n_ in norms]
    bp = ax_norm.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C9BE8")
        patch.set_alpha(0.6)
    ax_norm.set_ylabel("Token L2 norm", fontsize=9)
    ax_norm.set_title("Per-token L2 norm distribution per layer", fontsize=9)
    ax_norm.tick_params(axis="x", labelsize=7)
    ax_norm.grid(axis="y", alpha=0.3)

    # ── Row 3: layer-to-layer cosine similarity ──────────────
    ax_cos = fig.add_subplot(4, 1, 3)
    cos_sims = []
    cos_x = []
    for i in range(len(acts) - 1):
        sim = cosine_sim_layers(acts[i], acts[i + 1])
        cos_sims.append(sim)
        cos_x.append(f"{labels[i]}→{labels[i+1]}")
    if cos_sims:
        bars = ax_cos.bar(range(len(cos_sims)), cos_sims, color="#E87C4C", alpha=0.7)
        ax_cos.set_xticks(range(len(cos_sims)))
        ax_cos.set_xticklabels(cos_x, fontsize=7, rotation=30, ha="right")
        ax_cos.set_ylim(0, 1)
        ax_cos.set_ylabel("Mean cosine similarity", fontsize=9)
        ax_cos.set_title("Layer-to-layer similarity  (low = block changed tokens a lot)", fontsize=9)
        ax_cos.axhline(0.9, color="red", linestyle="--", linewidth=0.8, label="0.9 threshold")
        ax_cos.axhline(0.5, color="orange", linestyle="--", linewidth=0.8, label="0.5 threshold")
        ax_cos.legend(fontsize=7)
        ax_cos.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, cos_sims):
            ax_cos.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)

    # ── Row 4: variance explained by top-3 PCs ───────────────
    ax_var = fig.add_subplot(4, 1, 4)
    ve = [variance_explained(act, k=3) for act in acts]
    bars = ax_var.bar(range(n), [v * 100 for v in ve], color="#6DBF67", alpha=0.7)
    ax_var.set_xticks(range(n))
    ax_var.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax_var.set_ylabel("Variance explained (%)", fontsize=9)
    ax_var.set_title("Fraction of variance captured by top-3 PCs  (low = more spread / richer repr.)", fontsize=9)
    ax_var.set_ylim(0, 100)
    ax_var.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, ve):
        ax_var.text(bar.get_x() + bar.get_width() / 2, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Probe DiT context encoder layer representations.")
    p.add_argument("--config", default="configs/celeba256_sdvae_high_rama.yaml")
    p.add_argument("--checkpoint", default=None, help="path to .pt checkpoint with 'context_encoder' key")
    p.add_argument("--latents", default=None, help="latent cache directory (z_L/z_H .pt files)")
    p.add_argument("--num-images", type=int, default=16)
    p.add_argument("--out-dir", default="outputs/context_layer_probe")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    grid_size, patch_dim, patch_size, (C, H, W) = derive_grid_and_patch(config)
    # Context encoder sees z_L (half spatial res) and uses patch_size // 2
    ce_patch_size = patch_size // 2 or 1
    context_encoder_grid = (H // patch_size, W // patch_size)  # grid of context tokens

    enc_cfg = dict(config.get("context_encoder", {}))
    enc_cfg["patch_size"] = ce_patch_size
    enc_cfg["grid_size"] = list(context_encoder_grid)
    encoder = build_context_encoder(enc_cfg).to(args.device).eval()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        key = "context_encoder" if "context_encoder" in ckpt else None
        if key:
            encoder.load_state_dict(ckpt[key])
            print(f"Loaded encoder weights from {args.checkpoint}")
        else:
            print(f"  [warn] checkpoint has no 'context_encoder' key — using random weights")
    else:
        print("  [warn] no checkpoint provided — using random weights (probing untrained encoder)")

    arch = enc_cfg.get("architecture", "conv")
    if arch not in ("dit", "vit"):
        print(f"  [warn] architecture='{arch}' — layer hooks only capture the final block output for conv/resnet; DiT/ViT show per-block signals")

    print(f"\nConfig  : {args.config}")
    print(f"Arch    : {arch}")
    print(f"Device  : {args.device}")
    print(f"Grid    : {context_encoder_grid[0]}×{context_encoder_grid[1]} ({context_encoder_grid[0]*context_encoder_grid[1]} tokens)")

    z_l = get_latents(args, config, args.device)

    # Collect activations
    acts, labels = collect_layer_activations(encoder, z_l)
    norms, _ = collect_per_token_norms(encoder, z_l)

    print(f"\nCaptured {len(acts)} layer snapshots: {labels}")

    # Print quick text summary
    print("\nLayer summary:")
    print(f"  {'Layer':<12} {'Norm mean':>10} {'Norm std':>10} {'Var@3PC':>10} {'Cos→next':>10}")
    print("  " + "-" * 55)
    for i, (act, norm_vals, label) in enumerate(zip(acts, norms, labels)):
        ve = variance_explained(act) * 100
        cos = cosine_sim_layers(acts[i], acts[i + 1]) if i < len(acts) - 1 else float("nan")
        cos_str = f"{cos:.3f}" if not (cos != cos) else "  —  "
        print(f"  {label:<12} {norm_vals.mean().item():>10.3f} {norm_vals.std().item():>10.3f} {ve:>9.1f}% {cos_str:>10}")

    # Save figure
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer_probe.png"
    make_figure(acts, norms, labels, context_encoder_grid, out_path)

    # Interpretation hints
    print("\nHow to read the results:")
    print("  PCA maps    — smooth colour gradients mean spatial structure; noise means the layer isn't capturing it")
    print("  Norm boxplot— steadily growing norms are healthy; collapse to ~0 or explosion are bad signs")
    print("  Cos sim     — near 1.0 means the block barely changed the tokens (possibly redundant)")
    print("                near 0   means the block transformed them a lot (learning)")
    print("  Var@3PC     — high % means the representation is low-rank (collapsed); low % is richer")


if __name__ == "__main__":
    main()
