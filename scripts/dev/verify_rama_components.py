from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.rama import make_orthogonal_bases, patchify, unpatchify
from src.rama.projector import RAMAProjector
from src.rama.tokenizer import RAMATokenizer, build_tokenizer_from_config, load_tokenizer_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify RAMA patch, projection, and tokenizer components.")
    parser.add_argument("--bases", default="cache/rama_bases_p256_d16.pt")
    parser.add_argument("--tokenizer-config", default="cache/rama_tokenizer_config.pt")
    parser.add_argument("--num-patches", type=int, default=256)
    parser.add_argument("--patch-dim", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bases_path = Path(args.bases)
    bases = torch.load(bases_path, map_location="cpu").float() if bases_path.exists() else make_orthogonal_bases(
        args.num_patches,
        args.patch_dim,
    )
    bases = bases.to(args.device)
    projector = RAMAProjector(bases).to(args.device)

    z_h = torch.randn(4, 4, 32, 32, device=args.device)
    patches = patchify(z_h, patch_size=args.patch_size)
    z_h_rec = unpatchify(patches, channels=4, height=32, width=32, patch_size=args.patch_size)
    patch_err = (z_h - z_h_rec).abs().max().item()
    assert patch_err < 1e-6, f"patchify roundtrip error {patch_err}"

    i_hat = torch.einsum("pde,pdf->pef", bases, bases)
    eye = torch.eye(args.patch_dim, device=args.device)[None]
    ortho_err = (i_hat - eye).abs().max().item()
    assert ortho_err < 1e-5, f"orthogonality error {ortho_err}"

    patches_rec = projector.inverse(projector.project(patches))
    rama_err = (patches - patches_rec).abs().max().item()
    assert rama_err < 1e-5, f"RAMA roundtrip error {rama_err}"

    tokenizer_config = Path(args.tokenizer_config)
    tokenizer = (
        build_tokenizer_from_config(load_tokenizer_config(str(tokenizer_config)))
        if tokenizer_config.exists()
        else RAMATokenizer()
    )
    y = projector.project(patches)
    tokens = tokenizer.quantize(y)
    assert tokens.dtype == torch.long
    assert int(tokens.min()) >= 0
    assert int(tokens.max()) < tokenizer.num_bins
    y_hat = tokenizer.dequantize(tokens)
    assert y_hat.shape == y.shape

    print(f"patchify roundtrip max error: {patch_err:.8f}")
    print(f"orthogonality max error: {ortho_err:.8f}")
    print(f"RAMA roundtrip max error: {rama_err:.8f}")
    print(f"tokenizer sanity: shape={tuple(tokens.shape)} bins={tokenizer.num_bins}")


if __name__ == "__main__":
    main()

