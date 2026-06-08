# Flow Matching Model Setup for Latent-RAMA

This document defines the **macro flow matching model** used in the first Latent-RAMA experiment.

The goal is to train a flow matching model on the **low-frequency macro latent** \(z_L\), not directly on pixels and not on the full SD-VAE latent \(z\).

---

## 1. Problem Setup

We use a frozen SD-VAE encoder to map an image \(x\) into latent space:

\[
z = E(x)
\]

For CelebA 256 with Stable Diffusion VAE:

\[
x \in \mathbb{R}^{3 \times 256 \times 256}
\]

\[
z \in \mathbb{R}^{4 \times 32 \times 32}
\]

We then decompose the latent into macro and micro components:

\[
z_L = D_{\text{down}}(z)
\]

\[
z_H = z - U(z_L)
\]

where:

- \(z_L\): low-frequency macro latent
- \(z_H\): high-frequency residual latent
- \(D_{\text{down}}\): deterministic downsampling, e.g. average pooling
- \(U\): deterministic upsampling, e.g. bilinear interpolation

For the first setup:

\[
z_L \in \mathbb{R}^{4 \times 16 \times 16}
\]

The flow matching model learns:

\[
p(z_L)
\]

not:

\[
p(z)
\]

and not:

\[
p(z_H \mid z_L)
\]

The micro residual \(z_H\) is modeled separately by the RAMA micro model.

---

## 2. Training Target

The flow model learns a time-dependent vector field:

\[
v_\theta(z_t, t)
\]

that transports Gaussian noise to real macro latents.

Sample:

\[
z_0 \sim \mathcal{N}(0, I)
\]

\[
z_1 = z_L
\]

Define linear interpolation:

\[
z_t = (1-t)z_0 + t z_1
\]

where:

\[
t \sim \mathcal{U}(0,1)
\]

The target velocity is:

\[
v^\*(z_t,t) = z_1 - z_0
\]

The training loss is:

\[
\mathcal{L}_{FM}
=
\mathbb{E}_{z_0,z_1,t}
\left[
\left\|
v_\theta(z_t,t) - (z_1-z_0)
\right\|_2^2
\right]
\]

This is the standard rectified-flow / linear-path flow matching objective.

---

## 3. Data Pipeline

### 3.1 Input Images

Use CelebA-HQ or CelebA resized/cropped to:

```text
[3, 256, 256]
```

Recommended preprocessing:

```text
center crop
resize to 256
normalize to [-1, 1]
```

---

### 3.2 Encode with Frozen SD-VAE

Using Hugging Face diffusers:

```python
with torch.no_grad():
    posterior = vae.encode(images).latent_dist
    z = posterior.sample()
    z = z * vae.config.scaling_factor
```

Expected shape:

```text
[B, 4, 32, 32]
```

Important:

When decoding later:

```python
images = vae.decode(z / vae.config.scaling_factor).sample
```

Do not forget the VAE scaling factor.

---

### 3.3 Latent Decomposition

```python
import torch.nn.functional as F

def decompose_latent(z):
    z_l = F.avg_pool2d(z, kernel_size=2, stride=2)
    z_l_up = F.interpolate(
        z_l,
        size=z.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    z_h = z - z_l_up
    return z_l, z_h
```

For flow matching training, only use:

```python
z_l
```

Shape:

```text
[B, 4, 16, 16]
```

---

## 4. Recommended Latent Caching

Encoding images with SD-VAE during every training step is wasteful.

Recommended workflow:

```text
1. Load image batch.
2. Encode with frozen SD-VAE.
3. Save z as fp16 tensor.
4. During training, load cached z.
5. Compute z_L on the fly or cache z_L directly.
```

Storage estimate for 200k samples:

\[
200000 \times 4 \times 32 \times 32 \times 2 \text{ bytes}
\approx 1.64 \text{ GB}
\]

So caching full SD-VAE latents is cheap.

Recommended cache format:

```text
latents/
  000000.pt
  000001.pt
  ...
```

or a single memory-mapped array:

```text
celeba256_sdvae_latents_fp16.npy
```

---

## 5. Model Architecture

For the first experiment, use a small U-Net.

Input:

```text
z_t: [B, 4, 16, 16]
t:   [B]
```

Output:

```text
v_pred: [B, 4, 16, 16]
```

The model predicts the velocity field:

\[
v_\theta(z_t,t)
\]

---

## 6. Time Embedding

Use sinusoidal time embedding followed by an MLP.

```python
import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B], values in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb
```

Then:

```python
time_mlp = nn.Sequential(
    SinusoidalTimeEmbedding(128),
    nn.Linear(128, 512),
    nn.SiLU(),
    nn.Linear(512, 512),
)
```

---

## 7. Minimal U-Net Specification

Recommended first configuration:

```yaml
macro_flow_model:
  input_channels: 4
  output_channels: 4
  resolution: 16
  base_channels: 128
  channel_mult: [1, 2, 2]
  num_res_blocks: 2
  attention_resolutions: [8]
  time_embedding_dim: 512
  dropout: 0.0
```

This is enough for testing.

For stronger quality:

```yaml
macro_flow_model:
  input_channels: 4
  output_channels: 4
  resolution: 16
  base_channels: 192
  channel_mult: [1, 2, 4]
  num_res_blocks: 2
  attention_resolutions: [8, 4]
  time_embedding_dim: 768
  dropout: 0.0
```

---

## 8. Flow Matching Training Step

```python
import torch
import torch.nn.functional as F

def flow_matching_loss(model, z_l):
    # z_l: [B, 4, 16, 16]
    B = z_l.shape[0]
    device = z_l.device

    z0 = torch.randn_like(z_l)
    z1 = z_l

    t = torch.rand(B, device=device)
    t_view = t.view(B, 1, 1, 1)

    z_t = (1.0 - t_view) * z0 + t_view * z1
    target_v = z1 - z0

    pred_v = model(z_t, t)

    loss = F.mse_loss(pred_v, target_v)
    return loss
```

---

## 9. Training Loop

```python
for step, z in enumerate(dataloader):
    z = z.to(device)

    with torch.no_grad():
        z_l, _ = decompose_latent(z)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = flow_matching_loss(model, z_l)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    ema.update(model)

    if step % 100 == 0:
        print(f"step={step}, loss={loss.item():.6f}")
```

If using fp16 instead of bf16, use `GradScaler`.

---

## 10. Optimizer

Recommended:

```yaml
optimizer:
  type: AdamW
  lr: 2.0e-4
  betas: [0.9, 0.999]
  weight_decay: 0.0
  grad_clip: 1.0
```

For larger models:

```yaml
optimizer:
  type: AdamW
  lr: 1.0e-4
  betas: [0.9, 0.999]
  weight_decay: 1.0e-4
  grad_clip: 1.0
```

---

## 11. EMA

Use exponential moving average for sampling.

Recommended:

```yaml
ema:
  enabled: true
  decay: 0.9999
```

EMA model should be used for generation/evaluation.

---

## 12. Multi-GPU Setup

For two 16GB GPUs:

```yaml
distributed:
  backend: nccl
  num_gpus: 2
  precision: fp16_or_bf16
  per_gpu_batch_size: 64
  gradient_accumulation_steps: 1
  global_batch_size: 128
```

If memory allows:

```yaml
per_gpu_batch_size: 128
global_batch_size: 256
```

Effective batch size:

\[
B_{\text{global}}
=
B_{\text{per GPU}}
\times
N_{\text{GPU}}
\times
\text{grad accumulation}
\]

Example:

\[
64 \times 2 \times 2 = 256
\]

---

## 13. Sampling

At inference, sample:

\[
z_0 \sim \mathcal{N}(0,I)
\]

Then integrate:

\[
\frac{dz_t}{dt} = v_\theta(z_t,t)
\]

from \(t=0\) to \(t=1\).

---

### 13.1 Euler Sampler

```python
@torch.no_grad()
def sample_euler(model, shape, num_steps=50, device="cuda"):
    # shape: [B, 4, 16, 16]
    z = torch.randn(shape, device=device)
    B = shape[0]

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((B,), i / num_steps, device=device)
        v = model(z, t)
        z = z + dt * v

    return z
```

---

### 13.2 Heun Sampler

Heun is usually better than Euler for the same number of steps.

```python
@torch.no_grad()
def sample_heun(model, shape, num_steps=50, device="cuda"):
    z = torch.randn(shape, device=device)
    B = shape[0]
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((B,), i / num_steps, device=device)
        t_next = torch.full((B,), (i + 1) / num_steps, device=device)

        v = model(z, t)
        z_pred = z + dt * v

        v_next = model(z_pred, t_next)

        z = z + 0.5 * dt * (v + v_next)

    return z
```

Recommended first sampling steps:

```yaml
sampling:
  method: heun
  num_steps: 50
```

Also test:

```text
10, 20, 50, 100 steps
```

---

## 14. Combining with RAMA Later

The macro flow model only produces:

\[
\hat{z}_L \in \mathbb{R}^{4 \times 16 \times 16}
\]

For macro-only visualization:

\[
\hat{z} = U(\hat{z}_L)
\]

where:

\[
\hat{z} \in \mathbb{R}^{4 \times 32 \times 32}
\]

Then decode:

```python
z_l_up = F.interpolate(
    z_l_sampled,
    size=(32, 32),
    mode="bilinear",
    align_corners=False,
)

images = vae.decode(z_l_up / vae.config.scaling_factor).sample
```

For full Latent-RAMA generation:

\[
\hat{z} = U(\hat{z}_L) + \hat{z}_H
\]

where \(\hat{z}_H\) is sampled from the micro RAMA model.

---

## 15. Evaluation

Start with simple diagnostics.

### 15.1 Macro-only Decode

Decode:

\[
D(U(z_L))
\]

for real \(z_L\).

This shows how much information is preserved by the macro latent.

### 15.2 Macro Flow Samples

Generate:

\[
\hat{z}_L \sim p_\theta(z_L)
\]

then decode:

\[
D(U(\hat{z}_L))
\]

Expected result:

```text
coarse face structure
blurred details
weak high-frequency texture
```

This is normal.

### 15.3 FID

Compute FID on decoded images.

Compare:

```text
real images
VAE reconstructions
macro-only reconstructions
macro-flow samples
macro-flow + RAMA samples
```

---

## 16. Checkpoints

Recommended checkpoint contents:

```python
checkpoint = {
    "step": step,
    "model": model.state_dict(),
    "ema": ema.state_dict(),
    "optimizer": optimizer.state_dict(),
    "config": config,
}
```

Save every:

```yaml
checkpoint:
  every_steps: 10000
  keep_last: 5
```

---

## 17. Recommended First Config

```yaml
experiment:
  name: celeba256_sdvae_macro_flow
  image_size: 256
  latent_shape: [4, 32, 32]
  macro_shape: [4, 16, 16]

data:
  dataset: CelebA
  cache_latents: true
  latent_dtype: fp16

flow_matching:
  path: linear
  objective: velocity_prediction
  t_distribution: uniform_0_1

model:
  type: unet
  in_channels: 4
  out_channels: 4
  resolution: 16
  base_channels: 128
  channel_mult: [1, 2, 2]
  num_res_blocks: 2
  attention_resolutions: [8]
  time_embedding_dim: 512

training:
  precision: bf16
  num_gpus: 2
  per_gpu_batch_size: 64
  gradient_accumulation_steps: 2
  global_batch_size: 256
  optimizer: AdamW
  lr: 2.0e-4
  weight_decay: 0.0
  grad_clip: 1.0
  ema_decay: 0.9999
  max_steps: 400000

logging:
  tracker: wandb
  project: rama
  entity: null
  run_name: celeba256-sdvae-macro
  log_every_steps: 100
  sample_every_steps: 5000
  checkpoint_every_steps: 10000
  watch_model: false

sampling:
  solver: heun
  num_steps: 50
  use_ema: true
```

---

## 18. Expected Behavior

Early training:

```text
loss decreases quickly
macro-only samples look noisy or blob-like
```

Middle training:

```text
macro samples show coarse face layout
pose and face silhouette become visible
details remain blurry
```

Late training:

```text
stable face structure
better global consistency
still lacks high-frequency details
```

This is expected because the macro model only learns \(p(z_L)\).

The RAMA micro model is responsible for adding:

```text
skin texture
hair detail
sharp local residuals
edges
small identity-specific details
```

---

## 19. Failure Modes

### Failure 1: Decoded macro-only images are completely broken

Possible causes:

```text
wrong SD-VAE scaling factor
wrong image normalization
bad latent cache
incorrect upsampling shape
```

---

### Failure 2: Flow samples have exploding values

Possible causes:

```text
learning rate too high
no gradient clipping
bad mixed precision behavior
too few warmup steps
```

Try:

```yaml
lr: 1.0e-4
grad_clip: 1.0
precision: bf16
```

---

### Failure 3: Training loss decreases but samples are bad

Possible causes:

```text
model too small
too few training steps
Euler sampler too crude
EMA not used
```

Try:

```yaml
sampler: heun
num_steps: 100
ema_decay: 0.9999
base_channels: 192
```

---

### Failure 4: Macro samples look okay but final images lack details

This is expected for macro-only generation.

You need the RAMA micro model:

\[
\hat{z} = U(\hat{z}_L) + \hat{z}_H
\]

---

## 20. Minimal Milestones

Use these milestones before moving to RAMA.

```text
[ ] SD-VAE reconstruction works.
[ ] Latent cache works.
[ ] Macro-only reconstruction D(U(z_L)) is visually meaningful.
[ ] W&B run tracks loss, learning rate, gradient norm, EMA status, sample grids, and checkpoints.
[ ] Flow matching loss decreases.
[ ] Macro flow samples show coarse CelebA face structure.
[ ] Heun sampling works.
[ ] EMA sampling improves quality.
[ ] Macro-only baseline is saved for comparison.
```

Only after these are done should the micro RAMA model be trained and evaluated.
