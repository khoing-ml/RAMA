# Latent-RAMA First Setup: CelebA 256 + SD-VAE + Flow Matching

This document describes the first experimental setup for testing **Latent-RAMA Flow** on face images.

The goal is not to build the full final system immediately. The goal is to build a clean baseline pipeline where we can test:

1. Frozen SD-VAE latent extraction.
2. Macro/micro latent decomposition.
3. Flow matching on the macro latent.
4. Later: RAMA-based micro residual modeling.

---

## 1. Experiment Goal

We want to model images through a frozen latent space instead of pixel space.

Given an image:

\[
x \in \mathbb{R}^{3 \times 256 \times 256}
\]

A frozen Stable Diffusion VAE encoder maps it to a latent tensor:

\[
z = E(x) \in \mathbb{R}^{4 \times 32 \times 32}
\]

Then we decompose the latent into:

\[
z_L = D_{down}(z) \in \mathbb{R}^{4 \times 16 \times 16}
\]

\[
z_H = z - U(z_L) \in \mathbb{R}^{4 \times 32 \times 32}
\]

where:

- `z_L` is the **macro latent**.
- `z_H` is the **micro residual latent**.
- `D_down` is average pooling.
- `U` is bilinear upsampling.

The first baseline trains a flow matching model only on:

\[
z_L \in \mathbb{R}^{4 \times 16 \times 16}
\]

This gives a clean macro-generation baseline before adding the RAMA micro-generator.

---

## 2. Why This Setup?

CelebA 256 with SD-VAE is a good middle-ground experiment:

- Higher resolution than CIFAR-10.
- Much cheaper than full text-to-image training.
- SD-VAE gives a stable latent representation.
- Latent size is small enough for 2 GPUs with 16GB VRAM.
- The macro latent is only `4 x 16 x 16 = 1024` scalar dimensions.

The full SD-VAE latent has:

\[
4 \times 32 \times 32 = 4096
\]

scalar dimensions.

The macro latent has:

\[
4 \times 16 \times 16 = 1024
\]

scalar dimensions.

So the first flow matching model is small and easy to debug.

---

## 3. Hardware Assumption

Recommended setup:

```yaml
hardware:
  gpus: 2
  memory_per_gpu: 16GB
  precision: fp16 or bf16
```

This should be enough for:

- SD-VAE latent caching.
- Macro flow matching training.
- Micro RAMA experiments with small patch size.

For stable training, precompute and cache the SD-VAE latents instead of encoding images during every training step.

---

## 4. Dataset

Recommended dataset:

```yaml
dataset:
  name: CelebA-HQ or CelebA aligned/cropped
  image_size: 256
  channels: 3
  normalization: [-1, 1]
```

Images should be center-cropped or aligned and resized to:

```text
[3, 256, 256]
```

The VAE expects images normalized to approximately:

```text
[-1, 1]
```

---

## 5. Stable Diffusion VAE

Use the Stable Diffusion VAE checkpoint from Hugging Face:

```yaml
vae:
  provider: huggingface
  checkpoint_id: stabilityai/sd-vae-ft-mse
  cache_dir: .cache/huggingface
  dtype: fp16
  scaling_factor: read_from_config
```

Install the minimum Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision diffusers transformers accelerate safetensors huggingface_hub pillow tqdm
```

If the Hugging Face environment needs authentication:

```bash
hf auth login
```

Recommended loader:

```python
import torch
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    cache_dir=".cache/huggingface",
    torch_dtype=torch.float16,
).to("cuda")
vae.eval()
vae.requires_grad_(False)
```

This uses Hugging Face's Diffusers-format checkpoint directly. Prefer this path for the first setup because it automatically loads the VAE config, including `scaling_factor`.

If you want an explicit local snapshot first:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    local_dir="checkpoints/sd-vae-ft-mse",
)
```

Then load from disk:

```python
vae = AutoencoderKL.from_pretrained(
    "checkpoints/sd-vae-ft-mse",
    torch_dtype=torch.float16,
).to("cuda")
vae.eval()
vae.requires_grad_(False)
```

If you download a single VAE `.safetensors` or `.ckpt` file instead of a Diffusers-format folder, load it with `from_single_file`:

```python
vae = AutoencoderKL.from_single_file(
    "checkpoints/sd-vae-ft-mse/<downloaded-vae-file>.safetensors",
    torch_dtype=torch.float16,
).to("cuda")
vae.eval()
vae.requires_grad_(False)
```

For this project, the Diffusers-format Hugging Face repo is the preferred checkpoint source:

```text
stabilityai/sd-vae-ft-mse
```

After downloading with the `hf` CLI, the stable local path is:

```text
checkpoints/sd-vae-ft-mse
```

Encoding:

```python
with torch.no_grad():
    images = images.to(device="cuda", dtype=vae.dtype)
    posterior = vae.encode(images).latent_dist
    z = posterior.sample()
    z = z * vae.config.scaling_factor
```

Decoding:

```python
with torch.no_grad():
    images_hat = vae.decode(z / vae.config.scaling_factor).sample
```

Important: do not forget the VAE scaling factor.

The tensor passed to `vae.encode` must already be normalized to `[-1, 1]`, not `[0, 1]`.

---

## 6. Latent Decomposition

Given:

```python
z.shape == [B, 4, 32, 32]
```

Compute:

```python
import torch.nn.functional as F

z_L = F.avg_pool2d(z, kernel_size=2, stride=2)
z_L_up = F.interpolate(
    z_L,
    size=z.shape[-2:],
    mode="bilinear",
    align_corners=False,
)
z_H = z - z_L_up
```

Shapes:

```text
z:      [B, 4, 32, 32]
z_L:    [B, 4, 16, 16]
z_L_up: [B, 4, 32, 32]
z_H:    [B, 4, 32, 32]
```

---

## 7. Stage 0: VAE and Decomposition Sanity Checks

Before training any generative model, verify reconstruction quality.

### 7.1 VAE reconstruction

Decode the original latent:

\[
\hat{x}_{vae} = D(z)
\]

This is the upper-bound reconstruction quality.

### 7.2 Macro-only reconstruction

Decode only the upsampled macro latent:

\[
\hat{x}_{macro} = D(U(z_L))
\]

This shows how much image structure is kept in `z_L`.

### 7.3 Residual-only reconstruction

Decode only the high-frequency residual latent:

\[
\hat{x}_{H} = D(z_H)
\]

This shows what the residual component contributes on its own.

### 7.4 Full decomposition reconstruction

Decode:

\[
\hat{x}_{full} = D(U(z_L) + z_H)
\]

This should be numerically equivalent to VAE reconstruction, up to floating point error.

---

## 8. Stage 1: Cache SD-VAE Latents

Recommended latent cache format:

```text
latents/
  train_000000.pt
  train_000001.pt
  ...
```

Each file may store:

```python
{
    "z": z.half(),
    "z_L": z_L.half(),
    "z_H": z_H.half(),
}
```

However, the simplest cache stores only `z`:

```python
{
    "z": z.half()
}
```

Then compute `z_L` and `z_H` during training.

Estimated storage for 200k images in fp16:

\[
200000 \times 4 \times 32 \times 32 \times 2 \approx 1.64 \text{ GB}
\]

So latent caching is cheap.

---

## 9. Stage 2: Macro Flow Matching Baseline

Train flow matching on:

\[
z_L \in \mathbb{R}^{4 \times 16 \times 16}
\]

Sample Gaussian noise:

\[
z_0 \sim \mathcal{N}(0, I)
\]

Sample time:

\[
t \sim \mathcal{U}(0,1)
\]

Interpolate:

\[
z_t = (1-t)z_0 + t z_L
\]

Target velocity:

\[
v^* = z_L - z_0
\]

Train a model:

\[
v_\theta(z_t, t) \approx z_L - z_0
\]

Loss:

\[
\mathcal{L}_{macro}
=
\mathbb{E}_{t,z_0,z_L}
\left[
\|v_\theta(z_t,t) - (z_L-z_0)\|_2^2
\right]
\]

---

## 10. Macro Model Choice

Use a small U-Net first.

Recommended config:

```yaml
macro_flow_model:
  input_shape: [4, 16, 16]
  base_channels: 128
  channel_mult: [1, 2, 2]
  num_res_blocks: 2
  attention_resolutions: [8]
  time_embedding_dim: 256
  objective: rectified_flow
```

Avoid starting with a large DiT. A small U-Net is easier to debug.

---

## 11. Macro Flow Training Config

```yaml
training:
  batch_size_per_gpu: 64
  num_gpus: 2
  global_batch_size: 128
  gradient_accumulation_steps: 1
  optimizer: AdamW
  learning_rate: 2e-4
  weight_decay: 1e-4
  precision: fp16 or bf16
  ema: true
  ema_decay: 0.9999
  total_steps: 200000 to 500000

logging:
  tracker: wandb
  project: rama
  entity: null
  run_name: celeba256-sdvae-macro
  log_every_steps: 100
  sample_every_steps: 5000
  checkpoint_every_steps: 10000
  watch_model: false
```

If memory allows, increase batch size:

```yaml
batch_size_per_gpu: 128
global_batch_size: 256
```

---

## 12. Macro Sampling

Start from noise:

\[
z_0 \sim \mathcal{N}(0,I)
\]

Integrate the learned ODE:

\[
\frac{dz_t}{dt} = v_\theta(z_t,t)
\]

Euler sampler:

```python
@torch.no_grad()
def sample_macro(model, shape, steps=50, device="cuda"):
    z = torch.randn(shape, device=device)
    batch = shape[0]
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((batch,), i / steps, device=device)
        v = model(z, t)
        z = z + dt * v

    return z
```

Generated macro latent:

```text
z_L_sample: [B, 4, 16, 16]
```

For macro-only image generation:

```python
z_full = F.interpolate(
    z_L_sample,
    size=(32, 32),
    mode="bilinear",
    align_corners=False,
)
image = vae.decode(z_full / vae.config.scaling_factor).sample
```

This gives the first image-generation baseline.

---

## 13. Stage 3: Add RAMA Micro Residual Later

After macro flow works, add micro residual modeling.

Residual:

\[
z_H = z - U(z_L)
\]

Use patch size:

```yaml
rama:
  patch_size: 2
  latent_channels: 4
  patch_dim: 16
  num_patches: 256
```

Because:

\[
d = 2 \times 2 \times 4 = 16
\]

and:

\[
P = 16 \times 16 = 256
\]

For each patch:

\[
z_{H,p} \in \mathbb{R}^{16}
\]

Sample or cache a frozen orthogonal matrix:

\[
A_p \in \mathbb{R}^{16 \times 16}
\]

Project:

\[
y_{H,p} = A_p z_{H,p}
\]

Then either:

1. Quantize `y_{H,p,i}` and train categorical cross-entropy.
2. Use a 1D continuous density model such as spline flow.

For the first version, use categorical tokens.

---

## 14. First RAMA Micro Config

```yaml
micro_rama:
  patch_size: 2
  patch_dim: 16
  num_patches: 256
  orthogonal_basis: frozen_per_patch
  quantization_bins: 512
  quantization_bound: estimate_from_data_percentile_99_5

context_encoder:
  input: z_L
  output_grid: [16, 16]
  context_dim: 256
  architecture: shallow_cnn

micro_model:
  type: mlp
  input: context_vector + dimension_embedding
  dim_embedding: 64
  hidden_dim: 512
  num_layers: 4
  output_dim: 512
  loss: cross_entropy
```

---

## 15. Recommended Development Order

Do not train everything at once.

Follow this order:

```text
1. Load CelebA 256 images.
2. Encode with frozen SD-VAE.
3. Cache z latents.
4. Verify VAE reconstruction D(z).
5. Verify macro-only reconstruction D(U(z_L)).
6. Train macro flow matching on z_L with Weights & Biases logging enabled.
7. Sample macro-only images.
8. Add RAMA projection and quantization.
9. Test quantization-only reconstruction.
10. Train micro RAMA from ground-truth z_L.
11. Test ground-truth z_L + predicted z_H.
12. Combine sampled z_L + sampled z_H.
```

This order makes debugging much easier.

---

## 16. Evaluation Checklist

### Reconstruction diagnostics

- VAE reconstruction: `D(z)`
- Macro-only reconstruction: `D(U(z_L))`
- Full decomposition reconstruction: `D(U(z_L) + z_H)`
- Quantization-only RAMA reconstruction
- Predicted micro residual reconstruction

### Generation diagnostics

- Macro-only generated samples
- Macro + RAMA generated samples
- FID on CelebA validation split
- Visual diversity
- Face structure consistency
- Artifact level in high-frequency details

---

## 17. Expected Failure Modes

### 17.1 Macro flow produces blurry or distorted faces

Possible reasons:

- Macro model too small.
- Too few training steps.
- Bad VAE latent scaling.
- Learning rate too high.
- Sampler too few steps.

### 17.2 Macro-only images lack details

This is expected. `z_L` only contains low-frequency structure.

### 17.3 RAMA micro residual creates noise

Possible reasons:

- Conditional independence assumption is too strong.
- Quantization bound clips too much.
- Too few bins.
- Context encoder too weak.
- Patch size too small or too large.

### 17.4 Decoded images look extremely wrong

Likely causes:

- Forgot VAE scaling factor.
- Used wrong image normalization.
- Used unscaled latents for training but scaled latents for decoding.
- Mixed SD1.5 VAE and incompatible preprocessing.

---

## 18. Minimal Success Criteria

The first setup is successful if:

1. VAE reconstruction looks correct.
2. Macro-only reconstruction preserves face structure.
3. Macro flow can generate plausible low-frequency face latents.
4. Decoding `U(z_L_sample)` gives recognizable face-like samples.

The RAMA extension is successful if:

1. Quantization-only residual reconstruction is close to the original residual.
2. Predicted micro residual improves detail over macro-only reconstruction.
3. Full macro + RAMA generation improves visual sharpness without adding severe noise.

---

## 19. Initial Project Structure

```text
latent-rama-flow/
  configs/
    celeba256_sdvae_macro.yaml
    celeba256_sdvae_rama.yaml

  data/
    celeba256/
    latents/

  src/
    datasets/
      celeba.py
      latent_dataset.py

    models/
      unet_flow.py
      context_encoder.py
      micro_rama.py

    modules/
      vae_utils.py
      latent_decomposition.py
      flow_matching.py
      rama_projection.py
      quantization.py
      sampling.py

    train_cache_latents.py
    train_macro_flow.py
    train_micro_rama.py
    sample_macro.py
    sample_full.py

  notebooks/
    inspect_vae_reconstruction.ipynb
    inspect_macro_samples.ipynb
    inspect_rama_quantization.ipynb

  README.md
```

---

## 20. First Milestone

The first milestone should be:

```text
Generate macro-only CelebA 256 samples from SD-VAE latent flow matching.
```

That means:

1. Encode CelebA 256 to SD-VAE latents.
2. Downsample latents to `z_L`.
3. Train flow matching on `z_L`.
4. Sample `z_L` from Gaussian noise.
5. Upsample to `[4, 32, 32]`.
6. Decode with SD-VAE.
7. Inspect generated faces.

Only after this milestone works should RAMA micro residual modeling be added.
