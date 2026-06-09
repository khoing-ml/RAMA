# Discrete RAMA Implementation Gap Guide

This guide documents the missing implementation pieces required to align the repository with the **first-version discrete-token RAMA prototype** described in `docs/AGENTS.md`.

The current repository has mostly implemented the **continuous micro RAMA / nflows spline-flow route**, while the intended first prototype should use:

```text
RAMA projection
    -> quantization into discrete tokens
    -> categorical MicroRAMANet logits
    -> cross-entropy loss
```

The continuous `nflows` version is still useful, but it should be treated as an optional later variant.

---

## 1. Current Mismatch

The repository currently has these mismatches:

```text
1. No RAMAProjector class.
   Current repo only has free projection functions.

2. No RAMATokenizer class.
   Current repo only has helper quantization functions or no explicit tokenization module.

3. No quant-bound estimation script/function.
   Missing: scripts/estimate_quant_bound.py

4. No saved tokenizer config.
   Missing: cache/rama_tokenizer_config.pt

5. No quantization-only reconstruction script.
   Missing: scripts/test_quant_reconstruction.py

6. Micro training does not use quantization.
   Current train_micro_rama.py uses continuous nflows objective.

7. Current MicroRAMANet is continuous nflows spline model.
   The first prototype needs categorical logits [B, 256, 16, 256].

8. No categorical cross-entropy micro loss over tokens.

9. No scripts/sample_macro_flow.py.

10. No scripts/sample_full_model.py.

11. Macro sampler only has Heun.
    The guide asks for both Euler and Heun.

12. Required baseline outputs are missing:
    - quantization-only reconstruction
    - real-z_L micro reconstruction
    - full generated-z_L + sampled-z_H generation

13. No explicit verification tests for:
    - patchify roundtrip
    - RAMA inverse roundtrip
    - orthogonality
    - tiny-batch micro overfit
```

---

## 2. Main Correction

The first implementation should make the **discrete categorical RAMA model** the default.

The continuous `nflows` model should be kept, but moved behind an option:

```bash
--micro-type continuous
```

The default should be:

```bash
--micro-type categorical
```

Recommended naming:

```text
src/micro/micro_rama_categorical.py
src/micro/micro_rama_flow.py
```

or:

```text
src/micro/categorical_rama_net.py
src/micro/continuous_rama_flow.py
```

---

## 3. Target Micro Training Flow

The desired discrete-token training path is:

```text
cached latent z [B,4,32,32]
        ↓
decompose_latent
        ↓
z_L [B,4,16,16]       z_H [B,4,32,32]
        ↓                    ↓
ContextEncoder               patchify
        ↓                    ↓
context [B,256,256]          patches [B,256,16]
        ↓                    ↓
CategoricalMicroRAMANet      RAMAProjector.project
        ↓                    ↓
logits [B,256,16,V]          y [B,256,16]
                             ↓
                             RAMATokenizer.quantize
                             ↓
                             tokens [B,256,16]

loss = CrossEntropy(logits, tokens)
```

With the default first config:

```text
V = 256
P = 256
d = 16
```

So:

```text
logits: [B, 256, 16, 256]
tokens: [B, 256, 16]
```

---

## 4. Files to Add or Change

Required new files:

```text
src/rama/projector.py
src/rama/tokenizer.py
src/micro/micro_rama_categorical.py
scripts/estimate_quant_bound.py
scripts/test_quant_reconstruction.py
scripts/sample_macro_flow.py
scripts/sample_full_model.py
```

Required modified files:

```text
src/flow/sampler.py
src/micro/loss.py
scripts/train_micro_rama.py
configs/*.yaml
```

Optional rename:

```text
src/micro/micro_rama.py -> src/micro/micro_rama_flow.py
```

---

# 5. Add `RAMAProjector`

File:

```text
src/rama/projector.py
```

Purpose:

```text
Wrap RAMA projection and inverse projection in a module class.
Store bases as a non-trainable buffer.
```

Implementation:

```python
import torch
import torch.nn as nn


class RAMAProjector(nn.Module):
    def __init__(self, bases: torch.Tensor):
        super().__init__()

        assert bases.ndim == 3, f"Expected [P,d,d], got {bases.shape}"
        assert bases.shape[1] == bases.shape[2], f"Bases must be square, got {bases.shape}"

        self.register_buffer("bases", bases.float())

    @property
    def num_patches(self):
        return self.bases.shape[0]

    @property
    def patch_dim(self):
        return self.bases.shape[1]

    def project(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: [B, P, d]

        Returns:
            y: [B, P, d]
        """
        assert patches.ndim == 3
        assert patches.shape[1] == self.num_patches
        assert patches.shape[2] == self.patch_dim

        return torch.einsum("bpd,pde->bpe", patches, self.bases)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: [B, P, d]

        Returns:
            patches: [B, P, d]
        """
        assert y.ndim == 3
        assert y.shape[1] == self.num_patches
        assert y.shape[2] == self.patch_dim

        return torch.einsum("bpd,ped->bpe", y, self.bases)
```

Acceptance check:

```python
patches = torch.randn(4, 256, 16).to(device)
patches_rec = projector.inverse(projector.project(patches))
err = (patches - patches_rec).abs().max().item()
assert err < 1e-5
```

---

# 6. Add `RAMATokenizer`

File:

```text
src/rama/tokenizer.py
```

Purpose:

```text
Convert continuous RAMA coordinates y into discrete token ids.
Convert sampled token ids back into continuous RAMA coordinates.
```

Implementation:

```python
import torch
import torch.nn as nn


class RAMATokenizer(nn.Module):
    def __init__(self, num_bins: int = 256, bound: float = 3.0):
        super().__init__()

        assert num_bins > 1
        assert bound > 0

        self.num_bins = int(num_bins)
        self.bound = float(bound)

    def quantize(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: [B, P, d]

        Returns:
            tokens: [B, P, d], dtype torch.long
        """
        B = self.bound
        V = self.num_bins

        y = y.clamp(-B, B)
        y_norm = (y + B) / (2.0 * B)

        tokens = torch.floor(y_norm * V).long()
        tokens = tokens.clamp(0, V - 1)

        return tokens

    def dequantize(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, P, d]

        Returns:
            y_hat: [B, P, d]
        """
        assert tokens.dtype == torch.long

        B = self.bound
        V = self.num_bins

        y_hat = -B + (2.0 * B / V) * (tokens.float() + 0.5)
        return y_hat

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.quantize(y)

    def config_dict(self):
        return {
            "num_bins": self.num_bins,
            "bound": self.bound,
        }
```

Acceptance check:

```python
y = torch.randn(4, 256, 16)
tokens = tokenizer.quantize(y)
assert tokens.dtype == torch.long
assert tokens.shape == y.shape
assert tokens.min() >= 0
assert tokens.max() < tokenizer.num_bins

y_hat = tokenizer.dequantize(tokens)
assert y_hat.shape == y.shape
```

---

# 7. Add Quant-Bound Estimation

File:

```text
scripts/estimate_quant_bound.py
```

Purpose:

```text
Estimate the clipping bound B from RAMA-projected residual coordinates.
```

The bound should be:

```math
B = percentile(|y|, 99.5)
```

where:

```math
y = A_p z_{H,p}
```

Required CLI example:

```bash
python scripts/estimate_quant_bound.py \
  --latent-cache data/latents \
  --bases cache/rama_bases_p256_d16.pt \
  --output cache/rama_tokenizer_config.pt \
  --num-bins 256 \
  --percentile 99.5 \
  --max-batches 200
```

Core function:

```python
@torch.no_grad()
def estimate_quant_bound(
    dataloader,
    projector,
    patch_size=2,
    percentile=99.5,
    max_batches=200,
    device="cuda",
):
    values = []

    for step, z in enumerate(dataloader):
        if step >= max_batches:
            break

        z = z.to(device)

        z_l, z_h = decompose_latent(z)
        patches = patchify(z_h, patch_size=patch_size)
        y = projector.project(patches)

        values.append(y.abs().flatten().cpu())

    values = torch.cat(values, dim=0)
    bound = torch.quantile(values, percentile / 100.0).item()

    return bound
```

Save config:

```python
config = {
    "num_bins": args.num_bins,
    "bound": bound,
    "bound_method": "percentile_abs_y",
    "percentile": args.percentile,
    "patch_size": 2,
    "patch_dim": 16,
    "num_patches": 256,
}

torch.save(config, args.output)
```

Expected output:

```text
cache/rama_tokenizer_config.pt
```

---

# 8. Add Quantization-Only Reconstruction Script

File:

```text
scripts/test_quant_reconstruction.py
```

Purpose:

```text
Check whether quantization destroys too much high-frequency residual information.
```

The script should save a visual comparison grid:

```text
1. VAE reconstruction: D(z)
2. Macro-only reconstruction: D(U(z_L))
3. Quantization-only reconstruction: D(U(z_L) + quantized_z_H)
```

Core function:

```python
@torch.no_grad()
def quantization_only_reconstruct(
    z,
    projector,
    tokenizer,
    patch_size=2,
):
    z_l, z_h = decompose_latent(z)

    patches = patchify(z_h, patch_size=patch_size)
    y = projector.project(patches)

    tokens = tokenizer.quantize(y)
    y_hat = tokenizer.dequantize(tokens)

    patches_hat = projector.inverse(y_hat)

    z_h_hat = unpatchify(
        patches_hat,
        C=z.shape[1],
        H=z.shape[2],
        W=z.shape[3],
        patch_size=patch_size,
    )

    z_l_up = F.interpolate(
        z_l,
        size=z.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    z_hat = z_l_up + z_h_hat
    return z_hat
```

Acceptance criterion:

```text
Quantization-only reconstruction should be close to VAE reconstruction.
If it is bad, do not train MicroRAMANet yet.
```

Troubleshooting:

```text
1. Increase num_bins from 256 to 512.
2. Increase percentile from 99.5 to 99.9.
3. Check dequantization formula.
4. Check RAMA inverse projection.
5. Check patchify/unpatchify roundtrip.
```

---

# 9. Add Categorical Micro RAMA-Net

File:

```text
src/micro/micro_rama_categorical.py
```

Purpose:

```text
Predict categorical logits for quantized RAMA residual tokens.
```

Required class:

```python
import torch
import torch.nn as nn


class CategoricalMicroRAMANet(nn.Module):
    def __init__(
        self,
        context_dim=256,
        patch_dim=16,
        dim_emb_dim=64,
        hidden_dim=512,
        num_bins=256,
        num_layers=4,
    ):
        super().__init__()

        self.context_dim = context_dim
        self.patch_dim = patch_dim
        self.num_bins = num_bins

        self.dim_embed = nn.Embedding(patch_dim, dim_emb_dim)

        layers = []
        input_dim = context_dim + dim_emb_dim

        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, num_bins))

        self.net = nn.Sequential(*layers)

    def forward(self, context):
        """
        Args:
            context: [B, P, D_c]

        Returns:
            logits: [B, P, d, V]
        """
        B, P, D_c = context.shape
        d = self.patch_dim

        assert D_c == self.context_dim

        dim_ids = torch.arange(d, device=context.device)
        dim_emb = self.dim_embed(dim_ids)

        context = context[:, :, None, :].expand(B, P, d, D_c)
        dim_emb = dim_emb[None, None, :, :].expand(B, P, d, -1)

        x = torch.cat([context, dim_emb], dim=-1)
        logits = self.net(x)

        assert logits.shape == (B, P, d, self.num_bins)

        return logits
```

Expected output:

```text
logits: [B, 256, 16, 256]
```

---

# 10. Add Cross-Entropy Micro Loss

File:

```text
src/micro/loss.py
```

Add:

```python
import torch
import torch.nn.functional as F


def categorical_micro_loss(logits, tokens, num_bins):
    """
    Args:
        logits: [B, P, d, V]
        tokens: [B, P, d]

    Returns:
        scalar CE loss
    """
    assert tokens.dtype == torch.long
    assert logits.shape[:-1] == tokens.shape
    assert logits.shape[-1] == num_bins

    return F.cross_entropy(
        logits.reshape(-1, num_bins),
        tokens.reshape(-1),
    )
```

Keep the old continuous loss as:

```python
continuous_micro_nll_loss(...)
```

or:

```python
spline_micro_nll_loss(...)
```

Do not remove it.

---

# 11. Update Micro Training

File:

```text
scripts/train_micro_rama.py
```

Current training path:

```python
y = rama_project(patches, bases)
eps, logabsdet = micro_model(y, context)
loss = micro_nll_loss(eps, logabsdet)
```

This should become the optional continuous path.

Default categorical path should be:

```python
z_l, z_h = decompose_latent(z)

z_l = z_l.detach()
z_h = z_h.detach()

if context_noise_sigma > 0:
    z_l_input = z_l + context_noise_sigma * torch.randn_like(z_l)
else:
    z_l_input = z_l

patches = patchify(z_h, patch_size=2)
y = projector.project(patches)

tokens = tokenizer.quantize(y)

context = context_encoder(z_l_input)
logits = micro_model(context)

loss = categorical_micro_loss(
    logits=logits,
    tokens=tokens,
    num_bins=tokenizer.num_bins,
)
```

Required CLI flags:

```bash
--micro-type categorical
--micro-type continuous
--tokenizer-config cache/rama_tokenizer_config.pt
--bases cache/rama_bases_p256_d16.pt
```

Default:

```text
--micro-type categorical
```

Required behavior:

```text
If micro-type=categorical:
  load RAMATokenizer
  use CategoricalMicroRAMANet
  use cross-entropy loss

If micro-type=continuous:
  use existing nflows model
  use NLL/log-Jacobian loss
```

---

# 12. Add Euler Sampler

File:

```text
src/flow/sampler.py
```

Current repo has Heun only. Add Euler:

```python
@torch.no_grad()
def sample_euler(model, shape, num_steps=50, device="cuda"):
    z = torch.randn(shape, device=device)
    B = shape[0]
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((B,), i / num_steps, device=device)
        v = model(z, t)
        z = z + dt * v

    return z
```

Keep Heun as default.

---

# 13. Add Macro Sampling Script

File:

```text
scripts/sample_macro_flow.py
```

Purpose:

```text
Generate macro-only samples from the trained macro flow model.
```

Required behavior:

```text
1. Load macro flow checkpoint.
2. Use EMA weights if available.
3. Sample z_L_hat with Euler or Heun.
4. Upsample z_L_hat to [B, 4, 32, 32].
5. Decode with SD-VAE.
6. Save image grid.
```

Expected output:

```text
outputs/macro_samples/*.png
```

This baseline should look:

```text
face-like but blurry
```

---

# 14. Add Full Model Sampling Script

File:

```text
scripts/sample_full_model.py
```

Purpose:

```text
Generate images using macro flow + categorical micro RAMA.
```

Required behavior:

```text
1. Load macro flow checkpoint.
2. Load context encoder checkpoint.
3. Load categorical micro model checkpoint.
4. Load RAMA bases.
5. Load tokenizer config.
6. Sample z_L_hat using macro flow.
7. Compute context from z_L_hat.
8. Predict micro logits.
9. Sample tokens.
10. Dequantize tokens.
11. Inverse RAMA projection.
12. Unpatchify to z_H_hat.
13. Combine z_hat = U(z_L_hat) + z_H_hat.
14. Decode with SD-VAE.
15. Save image grid.
```

Sampling options:

```bash
--temperature 1.0
--use-argmax
--num-samples 64
--sampler heun
--steps 50
```

Default:

```text
categorical sampling, temperature 1.0
```

---

# 15. Add Verification Tests

Add a lightweight verification script:

```text
scripts/verify_rama_components.py
```

It should run these tests:

## 15.1 Patchify roundtrip

```python
z_h_rec = unpatchify(patchify(z_h))
assert (z_h - z_h_rec).abs().max() < 1e-6
```

## 15.2 Orthogonality

```python
I_hat = torch.einsum("pde,pdf->pef", bases, bases)
I = torch.eye(16, device=bases.device)[None]
assert (I_hat - I).abs().max() < 1e-5
```

## 15.3 RAMA inverse roundtrip

```python
patches_rec = projector.inverse(projector.project(patches))
assert (patches - patches_rec).abs().max() < 1e-5
```

## 15.4 Tokenizer sanity

```python
tokens = tokenizer.quantize(y)
assert tokens.dtype == torch.long
assert tokens.min() >= 0
assert tokens.max() < tokenizer.num_bins
y_hat = tokenizer.dequantize(tokens)
assert y_hat.shape == y.shape
```

---

# 16. Add Tiny-Batch Micro Overfit Test

Add:

```text
scripts/overfit_micro_tiny_batch.py
```

Purpose:

```text
Verify the categorical micro model can overfit 8-16 latents.
```

Required settings:

```yaml
num_images: 16
batch_size: 4
steps: 1000-3000
micro_type: categorical
disable_context_noise: true
use_argmax_reconstruction: true
```

Expected behavior:

```text
CE loss should drop strongly.
Argmax reconstruction should improve.
```

If this fails, likely bugs are:

```text
patch/context index mismatch
wrong loss reshape
wrong token dtype
bad quantization bound
RAMA inverse issue
patchify/unpatchify bug
```

---

# 17. Required Baseline Outputs

The repo should produce and save these outputs:

```text
outputs/vae_verification/
  original_vs_vae_recon.png

outputs/macro_samples/
  macro_only_samples.png

outputs/quantization_tests/
  vae_vs_macro_vs_quant.png

outputs/micro_reconstructions/
  real_zL_macro_vs_micro_argmax.png
  real_zL_macro_vs_micro_sampled.png

outputs/full_samples/
  generated_zL_macro_only.png
  generated_zL_macro_plus_micro.png
```

These are required before evaluating quality.

---

# 18. Updated Config Additions

Add categorical micro config:

```yaml
micro:
  type: categorical
  context_dim: 256
  patch_dim: 16
  dim_emb_dim: 64
  hidden_dim: 512
  num_layers: 4
  num_bins: 256
  loss: cross_entropy

tokenizer:
  num_bins: 256
  config_path: cache/rama_tokenizer_config.pt
  bound_method: percentile_abs_y
  percentile: 99.5

rama:
  bases_path: cache/rama_bases_p256_d16.pt
  patch_size: 2
  patch_dim: 16
  num_patches: 256
```

Keep continuous config separate:

```yaml
micro_continuous:
  type: nflows
  enabled: false
  loss: negative_log_likelihood
```

---

# 19. Migration Checklist

Use this checklist to track completion.

```text
[ ] Add RAMAProjector class.
[ ] Add RAMATokenizer class.
[ ] Add estimate_quant_bound.py.
[ ] Generate cache/rama_tokenizer_config.pt.
[ ] Add test_quant_reconstruction.py.
[ ] Add categorical MicroRAMANet.
[ ] Add categorical CE loss.
[ ] Update train_micro_rama.py to default to categorical mode.
[ ] Keep continuous nflows mode optional.
[ ] Add Euler sampler.
[ ] Add sample_macro_flow.py.
[ ] Add sample_full_model.py.
[ ] Add verify_rama_components.py.
[ ] Add overfit_micro_tiny_batch.py.
[ ] Save quantization-only reconstruction outputs.
[ ] Save real-z_L conditional reconstruction outputs.
[ ] Save full generated-z_L + sampled-z_H outputs.
```

---

# 20. Acceptance Criteria

The discrete RAMA prototype is acceptable only when all checks pass:

```text
1. RAMAProjector roundtrip error < 1e-5.
2. RAMATokenizer produces valid int64 tokens.
3. tokenizer config exists at cache/rama_tokenizer_config.pt.
4. quantization-only reconstruction looks close to VAE reconstruction.
5. categorical MicroRAMANet outputs [B, 256, 16, 256].
6. train_micro_rama.py uses CE loss in categorical mode.
7. tiny-batch overfit loss drops strongly.
8. sample_macro_flow.py produces macro-only samples.
9. sample_full_model.py produces macro + micro samples.
10. continuous nflows route remains optional, not default.
```

---

# 21. Do Not Do Yet

Do not add these before the categorical version works:

```text
1. Learned RAMA bases.
2. Patch size 4.
3. Transformer micro model.
4. End-to-end joint training.
5. Text conditioning.
6. CFG.
7. Large DiT macro model.
8. Making nflows the default again.
```

The immediate goal is to make the discrete-token RAMA path complete and testable.
