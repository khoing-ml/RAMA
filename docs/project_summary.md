# Latent-RAMA Project Summary

Latent-RAMA is a hybrid latent generative modeling idea. Instead of modeling an image directly in pixel space, the image is first encoded by a frozen autoencoder into a compact latent tensor. That latent is split into a low-frequency macro component and a high-frequency residual component:

```text
image x
  -> frozen autoencoder encoder
  -> full latent z
  -> macro latent z_L + residual latent z_H
```

The core factorization is:

```text
p(z) ~= p(z_L) p(z_H | z_L)
```

In this repository, the first concrete setup is:

```text
Dataset: CelebA 256
Autoencoder: Stable Diffusion VAE
Full latent z: [B, 4, 32, 32]
Macro latent z_L: [B, 4, 16, 16]
Residual latent z_H: [B, 4, 32, 32]
Residual patch size: 2
Residual patches: 256
Patch dimension: 16
```

## Why Split the Latent?

The macro latent is smaller and carries global structure:

```text
pose
layout
large shapes
coarse lighting
semantic organization
```

The residual latent carries local detail:

```text
edges
texture
fine identity details
local corrections
decoder-level high frequencies
```

This lets the project use a strong normal generative model for the global part and a lighter RAMA-based conditional model for local residual detail.

## Main Components

The current repo is organized around these components:

```text
1. Frozen SD-VAE
   Encodes images to latent z and decodes generated latents back to images.

2. Latent decomposition
   Computes z_L with average pooling and z_H = z - upsample(z_L).

3. Macro flow model
   Learns p(z_L) with rectified-flow / flow-matching training.

4. Context encoder
   Converts z_L into one context vector per residual patch.

5. RAMA projector
   Applies frozen per-patch orthogonal projections to residual patches.

6. RAMA tokenizer
   Quantizes projected residual coordinates into discrete tokens.

7. Categorical micro model
   Predicts token logits for each patch coordinate conditioned on z_L context.

8. Full sampler
   Samples z_L, samples or predicts z_H conditioned on z_L, reconstructs z, then decodes.
```

## Training Flow

The training flow is intentionally staged:

```text
cache images as SD-VAE latents
  -> train macro flow on z_L
  -> estimate RAMA quantization bound on z_H
  -> verify quantization-only reconstruction
  -> train micro RAMA model on z_H | z_L
  -> sample full model
```

The micro training target is:

```text
z
  -> decompose into z_L and z_H
  -> patchify z_H
  -> RAMA project residual patches
  -> quantize projected coordinates into tokens
  -> predict tokens from context(z_L)
```

With the default categorical setup:

```text
context: [B, 256, 256]
tokens:  [B, 256, 16]
logits:  [B, 256, 16, 256]
loss:    cross entropy
```

## Sampling Flow

At generation time:

```text
Gaussian noise
  -> macro flow sampler
  -> generated z_L
  -> context encoder
  -> categorical micro RAMA model
  -> sampled residual tokens
  -> dequantize tokens
  -> inverse RAMA projection
  -> unpatchify into z_H
  -> z = upsample(z_L) + z_H
  -> SD-VAE decoder
  -> generated image
```

## What Is Reusable?

The reusable idea is not tied to CelebA or SD-VAE. The portable pattern is:

```text
1. Choose a latent space.
2. Split it into macro and residual parts.
3. Train a global generator on the macro part.
4. Train a conditional local residual model on the residual part.
5. Recombine macro + residual before decoding or downstream prediction.
```

The macro model can be a U-Net flow, DiT, diffusion model, autoregressive transformer, VQ prior, or another architecture. The micro model can remain RAMA-based as long as the residual can be patchified into local vectors.

