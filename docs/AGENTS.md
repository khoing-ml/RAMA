# AGENTS.md

This file gives implementation instructions for AI coding agents working on the **Latent-RAMA** project.

The project goal is to implement and test a hybrid latent generative model:

```math
p(z) \approx p(z_L)p(z_H \mid z_L)
```

where:

- `z` is the full SD-VAE latent.
- `z_L` is the low-frequency macro latent.
- `z_H` is the high-frequency residual latent.
- The macro model is trained with flow matching.
- The micro model uses localized RAMA patchification and conditional residual modeling.

Target first experiment:

```text
Dataset: CelebA 256
Autoencoder: frozen Stable Diffusion VAE
Full latent z: [B, 4, 32, 32]
Macro latent z_L: [B, 4, 16, 16]
Residual latent z_H: [B, 4, 32, 32]
Patch size: 2
Patch dim: 16
Number of patches: 256
```

---

## 1. High-Level Implementation Order

Agents must implement the project in this order:

```text
1. SD-VAE encode/decode verification
2. Latent caching
3. Latent decomposition
4. Macro flow matching model
5. Macro sampler
6. Macro-only reconstruction and generation
7. Localized RAMA patchification
8. Cached orthogonal RAMA bases
9. RAMA projection and inverse projection
10. Discrete tokenization module
11. Quantization-only reconstruction test
12. Context encoder
13. Micro RAMA-Net
14. Micro training loop
15. Conditional reconstruction using real z_L
16. Full generation using generated z_L + sampled z_H
17. Baseline comparison
```

Do not skip verification steps.

---

## 2. Expected Repository Structure

Use this structure unless explicitly changed:

```text
latent-rama/
  configs/
    celeba256_sdvae_macro_flow.yaml
    celeba256_micro_rama.yaml
    celeba256_full_generation.yaml

  scripts/
    verify_vae.py
    cache_latents.py
    train_macro_flow.py
    sample_macro_flow.py
    estimate_quant_bound.py
    test_quant_reconstruction.py
    train_micro_rama.py
    sample_full_model.py

  src/
    data/
      celeba.py
      latent_dataset.py

    vae/
      sd_vae.py

    latent/
      decomposition.py

    flow/
      macro_unet.py
      time_embedding.py
      loss.py
      sampler.py
      ema.py

    rama/
      patchify.py
      bases.py
      projector.py
      tokenizer.py

    micro/
      context_encoder.py
      micro_rama_net.py
      loss.py

    utils/
      checkpoint.py
      image_grid.py
      seed.py
      distributed.py

  cache/
    rama_bases_p256_d16.pt
    rama_tokenizer_config.pt

  outputs/
    vae_verification/
    macro_samples/
    quantization_tests/
    micro_reconstructions/
    full_samples/

  README.md
  AGENTS.md
```

---

## 3. Core Tensor Shapes

Preserve these tensor shapes for the first version.

```text
z:        [B, 4, 32, 32]
z_L:      [B, 4, 16, 16]
z_H:      [B, 4, 32, 32]
patches:  [B, 256, 16]
bases:    [256, 16, 16]
y:        [B, 256, 16]
tokens:   [B, 256, 16]
context:  [B, 256, 256]
logits:   [B, 256, 16, V]
```

For the first version:

```text
V = 256
```

---

## 4. Latent Decomposition

Implement in:

```text
src/latent/decomposition.py
```

Required function:

```python
def decompose_latent(z):
    """
    Args:
        z: [B, 4, 32, 32]

    Returns:
        z_l: [B, 4, 16, 16]
        z_h: [B, 4, 32, 32]
    """
```

Definition:

```math
z_L = avgpool(z)
```

```math
z_H = z - upsample(z_L)
```

Use:

```python
avg_pool2d(kernel_size=2, stride=2)
interpolate(mode="bilinear", align_corners=False)
```

---

## 5. Frozen SD-VAE

Implement in:

```text
src/vae/sd_vae.py
```

Use:

```python
from diffusers import AutoencoderKL
```

Recommended checkpoint:

```text
stabilityai/sd-vae-ft-mse
```

Rules:

```text
1. VAE must be frozen.
2. VAE must be in eval mode.
3. Encode images normalized to [-1, 1].
4. Multiply encoded latents by vae.config.scaling_factor.
5. Divide latents by vae.config.scaling_factor before decoding.
```

Encoding:

```python
z = vae.encode(images).latent_dist.sample()
z = z * vae.config.scaling_factor
```

Decoding:

```python
images = vae.decode(z / vae.config.scaling_factor).sample
```

---

## 6. Macro Flow Matching Model

Implement in:

```text
src/flow/
```

Required modules:

```text
macro_unet.py
time_embedding.py
loss.py
sampler.py
ema.py
```

The macro model learns:

```math
p(z_L)
```

Use a small U-Net first.

Recommended config:

```yaml
macro_flow:
  network: SmallUNet
  input_channels: 4
  output_channels: 4
  resolution: 16
  base_channels: 128
  channel_mult: [1, 2, 2]
  num_res_blocks: 2
  attention_resolutions: [8]
  time_embedding_dim: 512
  objective: rectified_flow_velocity
```

Flow matching objective:

```math
z_0 \sim \mathcal{N}(0,I)
```

```math
z_1 = z_L
```

```math
z_t = (1-t)z_0 + tz_1
```

```math
v^* = z_1 - z_0
```

```math
\mathcal{L}_{macro}
=
\|v_\psi(z_t,t) - (z_1-z_0)\|^2
```

Required function:

```python
def flow_matching_loss(model, z_l):
    """
    Args:
        model: macro flow model
        z_l: [B, 4, 16, 16]

    Returns:
        scalar loss
    """
```

---

## 7. Macro Sampler

Implement in:

```text
src/flow/sampler.py
```

Required samplers:

```text
Euler
Heun
```

Input shape:

```text
[B, 4, 16, 16]
```

Output:

```text
sampled z_L: [B, 4, 16, 16]
```

Sampling ODE:

```math
\frac{dz_t}{dt} = v_\psi(z_t,t)
```

Default first config:

```yaml
sampler:
  method: heun
  num_steps: 50
```

---

## 8. Localized RAMA Patchification

Implement in:

```text
src/rama/patchify.py
```

Required functions:

```python
def patchify(z_h, patch_size=2):
    """
    Args:
        z_h: [B, 4, 32, 32]

    Returns:
        patches: [B, 256, 16]
    """
```

```python
def unpatchify(patches, C=4, H=32, W=32, patch_size=2):
    """
    Args:
        patches: [B, 256, 16]

    Returns:
        z_h: [B, 4, 32, 32]
    """
```

Mandatory sanity check:

```python
z_h_rec = unpatchify(patchify(z_h))
assert max_abs_error(z_h, z_h_rec) < 1e-6
```

---

## 9. Cached Orthogonal RAMA Bases

Implement in:

```text
src/rama/bases.py
```

Each patch position `p` has one frozen orthogonal matrix:

```math
A_p \in \mathbb{R}^{16 \times 16}
```

Shape:

```text
bases: [256, 16, 16]
```

Required function:

```python
def make_orthogonal_bases(num_patches=256, patch_dim=16, seed=1234):
    """
    Returns:
        bases: [256, 16, 16]
    """
```

Rules:

```text
1. Generate bases once.
2. Save them to cache/rama_bases_p256_d16.pt.
3. Reuse the same bases for training and inference.
4. Bases are not trainable.
5. Use float32 for bases.
```

Mandatory check:

```math
A_p^T A_p = I
```

---

## 10. RAMA Projector

Implement in:

```text
src/rama/projector.py
```

Required class:

```python
class RAMAProjector(nn.Module):
    def project(self, patches):
        """
        patches: [B, 256, 16]
        returns y: [B, 256, 16]
        """

    def inverse(self, y):
        """
        y: [B, 256, 16]
        returns patches: [B, 256, 16]
        """
```

Projection:

```math
y_{H,p} = A_p z_{H,p}
```

Inverse:

```math
z_{H,p} = A_p^T y_{H,p}
```

Mandatory check:

```python
patches_rec = projector.inverse(projector.project(patches))
assert max_abs_error(patches, patches_rec) < 1e-5
```

---

## 11. Discrete Tokenization Module

Implement in:

```text
src/rama/tokenizer.py
```

Required class:

```python
class RAMATokenizer(nn.Module):
    def quantize(self, y):
        """
        y: [B, 256, 16]
        returns tokens: [B, 256, 16], dtype long
        """

    def dequantize(self, tokens):
        """
        tokens: [B, 256, 16]
        returns y_hat: [B, 256, 16]
        """
```

Quantization:

```math
v = Q(y)
```

where:

```math
v \in \{0,\dots,V-1\}
```

Use:

```yaml
num_bins: 256
bound_method: percentile_abs_y
percentile: 99.5
```

Bound estimation function:

```python
def estimate_quant_bound(dataloader, projector, percentile=99.5):
    """
    Estimate B from |y| values.
    """
```

Save config:

```text
cache/rama_tokenizer_config.pt
```

Mandatory test:

```text
quantization-only reconstruction
```

Do not train the micro model until quantization-only reconstruction is acceptable.

---

## 12. Context Encoder

Implement in:

```text
src/micro/context_encoder.py
```

Network:

```text
Residual CNN with positional embedding
```

Input:

```text
z_L: [B, 4, 16, 16]
```

Output:

```text
context: [B, 256, 256]
```

Recommended config:

```yaml
context_encoder:
  input_channels: 4
  hidden_channels: 128
  context_dim: 256
  num_res_blocks: 4
  positional_embedding: true
  normalization: GroupNorm
  activation: SiLU
```

The context index `p` must match the residual patch index `p`.

---

## 13. Micro RAMA-Net

Implement in:

```text
src/micro/micro_rama_net.py
```

First version:

```text
Categorical MLP with coordinate embeddings
```

Input:

```text
context: [B, 256, 256]
```

Internal coordinate embedding:

```text
e_i: [16, 64]
```

Output:

```text
logits: [B, 256, 16, 256]
```

Recommended config:

```yaml
micro_rama_net:
  context_dim: 256
  patch_dim: 16
  dim_emb_dim: 64
  hidden_dim: 512
  num_layers: 4
  num_bins: 256
  normalization: LayerNorm
  activation: GELU
```

Training loss:

```python
loss = F.cross_entropy(
    logits.reshape(-1, num_bins),
    tokens.reshape(-1),
)
```

Tokens must be `torch.long`.

---

## 14. Optional Continuous Micro Model with nflows

Do not implement this first unless explicitly requested.

The continuous version replaces categorical token prediction with a conditional 1D Rational-Quadratic Spline Flow.

Library:

```text
nflows
```

Target:

```math
p(y_{H,p,i} \mid c_p,e_i)
```

Loss:

```math
-\log p(y_{H,p,i} \mid c_p,e_i)
```

Use after the categorical version works.

---

## 15. Training Rules

Use separate optimizers.

### Macro optimizer

Updates:

```text
macro_flow_model
```

Does not update:

```text
context_encoder
micro_rama_net
SD-VAE
RAMA bases
```

### Micro optimizer

Updates:

```text
context_encoder
micro_rama_net
```

Does not update:

```text
macro_flow_model
SD-VAE
RAMA bases
```

During micro training:

```python
z_l = z_l.detach()
z_h = z_h.detach()
```

Optional robustness:

```python
z_l_input = z_l + 0.03 * torch.randn_like(z_l)
```

---

## 16. Required Verification Scripts

Agents must create these scripts.

### `scripts/verify_vae.py`

Output grid:

```text
original image
VAE reconstruction
```

### `scripts/cache_latents.py`

Input:

```text
CelebA images
```

Output:

```text
cached latents z: [4, 32, 32]
```

### `scripts/sample_macro_flow.py`

Output:

```text
macro-only generated images: D(U(z_L_hat))
```

Expected:

```text
face-like but blurry
```

### `scripts/test_quant_reconstruction.py`

Output grid:

```text
VAE reconstruction: D(z)
macro-only reconstruction: D(U(z_L))
quantization reconstruction: D(U(z_L) + quantized_z_H)
```

### `scripts/train_micro_rama.py`

Trains:

```text
context_encoder + micro_rama_net
```

### `scripts/sample_full_model.py`

Generates:

```text
z_L_hat from macro flow
z_H_hat from micro RAMA
z_hat = U(z_L_hat) + z_H_hat
image = SD-VAE.decode(z_hat)
```

---

## 17. Required Baselines

Preserve outputs for:

```text
A. VAE reconstruction
B. Macro-only reconstruction
C. Macro-only flow generation
D. Quantization-only reconstruction
E. Real-z_L conditional micro reconstruction
F. Full generated-z_L + sampled-z_H generation
```

Later compare against:

```text
G. Full latent flow trained directly on z: [4, 32, 32]
```

---

## 18. Success Criteria

The first version is successful only if:

```text
1. SD-VAE reconstruction works.
2. Macro-only reconstruction is meaningful.
3. Macro flow samples are face-like but blurry.
4. Patchify/unpatchify is exact.
5. RAMA project/inverse is near exact.
6. Quantization-only reconstruction is acceptable.
7. Micro model can overfit a tiny batch.
8. Real-z_L + predicted z_H improves over macro-only reconstruction.
9. Generated-z_L + sampled z_H is sharper than macro-only generation.
```

Do not claim success before these checks pass.

---

## 19. Coding Standards

Follow these rules:

```text
1. Keep tensor shapes explicit in comments.
2. Add assertions for important shapes.
3. Keep macro and micro training separate.
4. Do not train the SD-VAE.
5. Do not make RAMA bases trainable.
6. Save all configs with checkpoints.
7. Use deterministic seeds for bases and experiments.
8. Use bf16 if supported; otherwise fp16 with GradScaler.
9. Use gradient clipping.
10. Log visual samples regularly.
```

---

## 20. First Config Summary

```yaml
experiment:
  name: celeba256_latent_rama_v1

vae:
  checkpoint: stabilityai/sd-vae-ft-mse
  trainable: false
  latent_shape: [4, 32, 32]

decomposition:
  macro_shape: [4, 16, 16]
  residual_shape: [4, 32, 32]
  downsample: avg_pool2d
  upsample: bilinear

macro_flow:
  model: SmallUNet
  base_channels: 128
  channel_mult: [1, 2, 2]
  attention_resolutions: [8]
  time_embedding_dim: 512
  lr: 2.0e-4
  ema_decay: 0.9999
  sampler: heun
  sampling_steps: 50

rama:
  patch_size: 2
  patch_dim: 16
  num_patches: 256
  bases_shape: [256, 16, 16]
  bases_seed: 1234

tokenizer:
  num_bins: 256
  bound_percentile: 99.5

context_encoder:
  model: ResidualCNN
  hidden_channels: 128
  context_dim: 256
  num_res_blocks: 4
  positional_embedding: true

micro_rama_net:
  model: CategoricalMLP
  dim_emb_dim: 64
  hidden_dim: 512
  num_layers: 4
  num_bins: 256

training:
  precision: bf16
  num_gpus: 2
  batch_per_gpu: 64
  grad_clip: 1.0
```

---

## 21. Do Not Do These Yet

Do not implement these until the first prototype works:

```text
1. End-to-end joint training.
2. Learned RAMA bases.
3. Patch size 4 or larger.
4. nflows continuous RAMA model.
5. Transformer micro-generator.
6. Text conditioning.
7. Classifier-free guidance.
8. Large DiT macro generator.
```

First make the simple version work.
