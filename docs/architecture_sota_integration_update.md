# Update: Neural Network Architecture & SOTA Integration

This update adds the **Neural Network Architecture & SOTA Integration** part to the Latent-RAMA implementation plan.

The purpose of this update is to clarify how the full system should be organized as a modular architecture:

```text
SD-VAE latent space
      ↓
macro / micro decomposition
      ↓
macro flow matching model
      ↓
context encoder
      ↓
micro RAMA-Net
      ↓
SD-VAE decoder
```

The key design principle is:

> Use a standard generative model for the low-frequency macro latent, and use a lightweight RAMA-based model for the high-frequency residual.

---

## 1. Full Architecture Overview

Given an image \(x\), a frozen SD-VAE encoder maps it into latent space:

\[
z = E(x)
\]

For CelebA 256:

\[
x \in \mathbb{R}^{3 \times 256 \times 256}
\]

\[
z \in \mathbb{R}^{4 \times 32 \times 32}
\]

Then decompose:

\[
z_L = D_{\text{down}}(z)
\]

\[
z_H = z - U(z_L)
\]

where:

```text
z_L: [4, 16, 16]  macro latent
z_H: [4, 32, 32]  high-frequency residual latent
```

The model factorizes generation as:

\[
p(z) \approx p(z_L)p(z_H \mid z_L)
\]

The architecture has three trainable components:

```text
1. Macro generator       v_ψ
2. Context encoder       E_θ
3. Micro RAMA-Net        M_φ
```

---

## 2. Component B.1: Macro Generator

The macro generator models:

\[
p(z_L)
\]

where:

\[
z_L \in \mathbb{R}^{4 \times 16 \times 16}
\]

This component can be a standard generative architecture:

```text
U-Net flow matching model
DiT
Rectified Flow model
Flux-style flow model
latent diffusion model
```

For the first implementation, use a **small U-Net flow matching model**.

---

### 2.1 Macro Flow Matching Objective

Sample Gaussian noise:

\[
z_0 \sim \mathcal{N}(0,I)
\]

Let:

\[
z_1 = z_L
\]

Sample time:

\[
t \sim \mathcal{U}(0,1)
\]

Interpolate:

\[
z_t = (1-t)z_0 + tz_1
\]

Target velocity:

\[
v^\* = z_1 - z_0
\]

Train:

\[
\mathcal{L}_{macro}
=
\left\|
v_\psi(z_t,t) - (z_1-z_0)
\right\|^2
\]

This model is responsible for global structure:

```text
face pose
coarse face shape
hair mass
background layout
lighting structure
large semantic organization
```

---

### 2.2 Macro Generator Input and Output

```text
input:
  z_t: [B, 4, 16, 16]
  t:   [B]

output:
  velocity: [B, 4, 16, 16]
```

Recommended first config:

```yaml
macro_generator:
  type: unet_flow_matching
  target: z_L
  input_shape: [4, 16, 16]
  output_shape: [4, 16, 16]
  base_channels: 128
  channel_mult: [1, 2, 2]
  num_res_blocks: 2
  attention_resolutions: [8]
  time_embedding_dim: 512
  objective: velocity_mse
```

---

## 3. Component B.2: Context Encoder

The context encoder maps the macro latent \(z_L\) to local conditioning vectors.

\[
C = E_\theta(z_L)
\]

For the first setup:

```text
z_L:     [B, 4, 16, 16]
context: [B, 256, D_c]
```

Since the residual patch grid is \(16 \times 16\), the context encoder produces one context vector for each residual patch.

\[
C = \{c_p\}_{p=1}^{256}
\]

where:

\[
c_p \in \mathbb{R}^{D_c}
\]

Recommended:

\[
D_c = 256
\]

---

### 3.1 Context Encoder Role

The context encoder tells the micro model what local detail should be generated.

Examples:

```text
macro region from z_L       expected micro residual
---------------------------------------------------
eye region                  iris boundary, eyelash detail, sharp edges
hair region                 hair strands and texture
skin region                 smooth local texture
mouth region                lip boundary and teeth detail
background region           weak or simple residual
```

Without the context encoder, the micro model would learn only a generic residual distribution:

\[
p(z_H)
\]

With the context encoder, it learns:

\[
p(z_H \mid z_L)
\]

---

### 3.2 Recommended Context Encoder

Use a shallow CNN with positional embedding.

```yaml
context_encoder:
  type: shallow_cnn_with_pos_emb
  input_shape: [4, 16, 16]
  output_grid: [16, 16]
  output_shape: [256, 256]
  input_channels: 4
  hidden_channels: 128
  context_dim: 256
  num_layers: 3
  normalization: groupnorm
  activation: silu
  positional_embedding: true
  context_noise_sigma: 0.03
```

The positional embedding is useful because CelebA faces are roughly aligned.

---

## 4. Component B.3: Micro-Generator: 1D RAMA-Net

The micro RAMA-Net models the high-frequency residual:

\[
p(z_H \mid z_L)
\]

Use a batch-parallelized MLP acting as a **Conditional Rational-Quadratic Neural Spline Flow (RQ-NSF)**. In code, this component is built with the `nflows` PyTorch spline implementation.

First, patchify \(z_H\):

\[
z_H \rightarrow \{z_{H,p}\}_{p=1}^{P}
\]

For patch size \(2 \times 2\):

```text
z_H:     [B, 4, 32, 32]
patches: [B, 256, 16]
```

Each patch vector is:

\[
z_{H,p} \in \mathbb{R}^{16}
\]

Then apply the frozen RAMA projection:

\[
y_{H,p} = A_p z_{H,p}
\]

where:

\[
A_p \in \mathbb{R}^{16 \times 16}
\]

and:

```text
bases: [256, 16, 16]
```

---

### 4.1 Continuous Spline-Flow Micro Modeling

Each scalar \(y_{H,p,i}\) is transformed by a conditional monotonic spline:

\[
f_\phi: y_{H,p,i} \rightarrow \epsilon_{p,i}
\]

where:

\[
\epsilon_{p,i} \sim \mathcal{N}(0,1)
\]

The spline parameters are predicted from:

```text
y_H,p,i: scalar projected residual coordinate
c_p:     local context vector from the context encoder
e_i:     learned coordinate embedding for dimension index i
```

The input layer concatenates:

\[
[y_{H,p,i}, c_p, e_i]
\]

The hidden layers use residual fully connected blocks with LayerNorm and Swish/SiLU activations.

The output layer predicts a vector of size:

\[
3B - 1
\]

where \(B\) is the number of spline bins. These parameters define widths, heights, and derivatives for the 1D rational-quadratic spline.

The micro loss is:

\[
\mathcal{L}_{micro}
=
\sum_{p=1}^{P}
\sum_{i=1}^{d}
-\log p_\phi(y_{H,p,i} \mid c_p,e_i)
\]

---

### 4.2 Micro RAMA-Net Shape

For the first setup:

```text
context: [B, 256, 256]
input y: [B, 256, 16]
eps:     [B, 256, 16]
logdet:  [B, 256, 16]
```

Recommended config:

```yaml
micro_rama_net:
  type: conditional_rq_nsf
  context_dim: 256
  patch_dim: 16
  dim_emb_dim: 64
  hidden_dim: 512
  num_layers: 4
  spline_bins: 16
  tail_bound: 3.0
  normalization: layernorm
  activation: silu
```

---

## 5. How the Components Connect

### Training-Time Flow

During training, use real SD-VAE latents:

```text
image x
  ↓
frozen SD-VAE encoder
  ↓
z
  ↓
z_L = downsample(z)
z_H = z - upsample(z_L)
```

Then train two objectives separately.

---

### 5.1 Macro Update

```text
input:
  real z_L

train:
  macro flow model v_ψ

loss:
  flow matching MSE
```

Only update:

```text
v_ψ
```

---

### 5.2 Micro Update

```text
input:
  real z_L
  real z_H

process:
  z_H → patchify → RAMA projection
  z_L → context encoder
  y_H,p,i + context + dimension embedding → micro RAMA-Net

loss:
  negative log likelihood under conditional RQ-NSF
```

Only update:

```text
E_θ
M_φ
```

Do not update:

```text
SD-VAE
macro flow model
RAMA bases
```

---

## 6. Inference-Time Flow

At generation time:

```text
1. Sample z_L using macro flow model.
2. Encode z_L into local context C.
3. Sample standard-normal noise for each projected residual coordinate.
4. Invert the conditional spline flow to obtain RAMA coordinates.
5. Apply inverse RAMA projection.
6. Unpatchify into z_H.
7. Combine z = U(z_L) + z_H.
8. Decode with SD-VAE.
```

Mathematically:

\[
\hat{z}_L \sim p_\psi(z_L)
\]

\[
C = E_\theta(\hat{z}_L)
\]

\[
\epsilon_{p,i} \sim \mathcal{N}(0,1)
\]

\[
\hat{y}_{H,p,i} = f_\phi^{-1}(\epsilon_{p,i}; c_p,e_i)
\]

\[
\hat{z}_{H,p} = A_p^T \hat{y}_{H,p}
\]

\[
\hat{z} = U(\hat{z}_L) + \hat{z}_H
\]

\[
\hat{x} = D(\hat{z})
\]

---

## 7. Why This Counts as SOTA Integration

The method does not require building a full image generator from scratch.

Instead, it reuses strong existing components:

```text
autoencoder:
  SD-VAE or VQGAN

macro generator:
  DiT / U-Net / Flow Matching / Rectified Flow

micro generator:
  lightweight RAMA-Net
```

The main architectural novelty is the decomposition:

\[
p(z) \approx p(z_L)p(z_H \mid z_L)
\]

and the RAMA approximation:

\[
p(z_H \mid z_L)
\approx
\prod_{p=1}^{P}
\prod_{i=1}^{d}
p_\phi(y_{H,p,i} \mid c_p,e_i)
\]

So the macro model can remain close to standard SOTA designs.

---

## 8. Advantages of This Architecture

Compared with full latent flow matching on:

\[
z \in \mathbb{R}^{4 \times 32 \times 32}
\]

this architecture trains the flow model only on:

\[
z_L \in \mathbb{R}^{4 \times 16 \times 16}
\]

Benefits:

```text
smaller flow target
cheaper macro ODE sampling
one-shot micro residual generation
modular debugging
separate global structure and local detail
easy ablations
```

Expected behavior:

```text
macro flow:
  learns global face structure

micro RAMA-Net:
  adds local texture and sharp residual detail
```

---

## 9. Important Risks

The method depends on a strong conditional independence approximation:

\[
p(z_H \mid z_L)
\approx
\prod_{p,i}
p(y_{H,p,i} \mid c_p,e_i)
\]

This may fail if \(z_H\) still contains structured global information.

Possible failure cases:

```text
identity-specific details are lost
eyes or mouth become inconsistent
hair texture becomes noisy
micro residual adds artifacts
generated z_L causes distribution shift
```

The method is successful only if:

```text
z_L captures most semantic structure
z_H mostly contains local residual detail
RAMA projection makes dimensions easier to model
quantization does not destroy too much information
micro model improves macro-only reconstruction
```

---

## 10. Updated Implementation Plan

Recommended implementation order:

```text
1. Verify SD-VAE reconstruction.
2. Verify macro-only reconstruction D(U(z_L)).
3. Cache SD-VAE latents.
4. Train macro flow on z_L.
5. Generate macro-only samples.
6. Create cached RAMA spatial bases.
7. Test patchify/unpatchify.
8. Test RAMA project/inverse project.
9. Estimate spline tail bound B.
10. Test flow-only likelihood/reconstruction sanity checks.
11. Train context encoder + micro RAMA-Net on real z_L, z_H.
12. Test conditional reconstruction using real z_L.
13. Combine generated z_L with sampled z_H.
14. Compare against full latent flow baseline.
```

---

## 11. Updated First Experiment Config

```yaml
experiment:
  name: latent_rama_sota_integration_celeba256
  dataset: CelebA
  image_size: 256
  vae: frozen_sd_vae

latent:
  full_shape: [4, 32, 32]
  macro_shape: [4, 16, 16]
  residual_shape: [4, 32, 32]
  downsample: avg_pool_2d
  upsample: bilinear

macro_generator:
  type: unet_flow_matching
  objective: velocity_mse
  input_shape: [4, 16, 16]
  output_shape: [4, 16, 16]
  base_channels: 128
  channel_mult: [1, 2, 2]
  attention_resolutions: [8]
  sampling_solver: heun
  sampling_steps: 50

context_encoder:
  type: shallow_cnn_with_pos_emb
  input_shape: [4, 16, 16]
  output_grid: [16, 16]
  context_dim: 256
  hidden_channels: 128
  context_noise_sigma: 0.03

micro_rama:
  patch_size: 2
  patch_grid: [16, 16]
  num_patches: 256
  patch_dim: 16
  basis_type: frozen_orthogonal
  basis_shape: [256, 16, 16]
  spline_tail_bound: percentile_abs_y_99_5

micro_rama_net:
  type: conditional_rq_nsf
  context_dim: 256
  dim_emb_dim: 64
  hidden_dim: 512
  num_layers: 4
  spline_bins: 16
  tail_bound: 3.0

training:
  precision: bf16
  num_gpus: 2
  macro_lr: 2.0e-4
  micro_lr: 2.0e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0
```

---

## 12. Baselines for Comparison

To evaluate the method, compare:

```text
A. Full latent flow:
   train flow matching on z: [4, 32, 32]

B. Macro-only flow:
   train flow matching on z_L: [4, 16, 16]
   decode U(z_L)

C. Latent-RAMA hybrid:
   train macro flow on z_L
   train RAMA micro model on z_H | z_L
   decode U(z_L) + z_H
```

The method is useful if:

```text
C is much better than B
C is competitive with A
C is faster or cheaper than A
```

Evaluation metrics:

```text
FID
sampling time
training memory
visual quality
macro-only vs macro+micro improvement
conditional reconstruction quality
```

---

## 13. Summary

This update defines the architecture as a modular SOTA-compatible system:

\[
\text{SD-VAE}
+
\text{Macro Flow Matching}
+
\text{Context Encoder}
+
\text{Micro RAMA-Net}
\]

The macro generator handles:

```text
global structure
semantic layout
coarse latent distribution
```

The micro RAMA-Net handles:

```text
local residuals
texture
high-frequency detail
```

The context encoder connects them by producing local conditions:

\[
c_p = E_\theta(z_L)_p
\]

The final generated latent is:

\[
\hat{z} = U(\hat{z}_L) + \hat{z}_H
\]

and the generated image is:

\[
\hat{x} = D(\hat{z})
\]
