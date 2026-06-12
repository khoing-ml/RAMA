# Portable Latent-RAMA Architecture Guide

This guide explains how to reuse the Latent-RAMA idea in another repository or with another architecture.

The key principle is:

```text
Separate global generation from local residual generation.
```

Do not copy every file blindly. Copy the interface and tensor contracts first, then adapt the model classes to the host repository.

## 1. Identify the Host Latent Space

Start by choosing the representation that the new architecture should model.

Examples:

```text
SD-VAE image latent:        [B, 4, 32, 32]
VQGAN continuous latent:    [B, C, H, W]
Video latent:               [B, C, T, H, W]
Audio spectrogram latent:   [B, C, F, T]
Feature-map latent:         [B, C, H, W]
Token embedding sequence:   [B, N, D]
```

The only requirement is that you can define:

```text
full representation = macro representation + residual representation
```

or an equivalent deterministic reconstruction rule.

## 2. Define Macro and Residual Decomposition

For image-like latents, the simplest decomposition is:

```text
z_L = downsample(z)
z_H = z - upsample(z_L)
z   = upsample(z_L) + z_H
```

In this repo:

```text
z:   [B, 4, 32, 32]
z_L: [B, 4, 16, 16]
z_H: [B, 4, 32, 32]
```

For another architecture, choose a decomposition that preserves an exact or near-exact reconstruction path:

```text
2D images:       average pool + bilinear upsample
video:           spatial or spatiotemporal downsample + upsample
audio:           frequency/time downsample + upsample
sequences:       coarse tokens + residual token embeddings
3D volumes:      trilinear downsample + upsample
```

Keep this module small and heavily tested. Everything else depends on its shape contract.

## 3. Choose the Macro Generator

The macro generator models:

```text
p(z_L)
```

It is responsible for the coarse structure. The current repo uses a small U-Net flow-matching model, but this component is intentionally replaceable.

Possible macro generators:

```text
U-Net flow matching model
DDPM or latent diffusion model
DiT / transformer flow model
autoregressive token prior
masked token model
normalizing flow
GAN-style latent generator
```

The macro generator only needs to provide one interface:

```python
sample_macro(batch_size) -> z_L
```

For training, it needs a loss against real macro latents:

```python
macro_loss(model, z_L)
```

In the current repo, the training target is rectified flow:

```text
z_0 ~ Normal(0, I)
z_1 = z_L
z_t = (1 - t) z_0 + t z_1
target velocity = z_1 - z_0
```

## 4. Build the Residual Patch Interface

The micro model works on local residual vectors.

For a 2D latent:

```text
z_H:     [B, C, H, W]
patches: [B, P, d]
```

where:

```text
P = (H / patch_size) * (W / patch_size)
d = C * patch_size * patch_size
```

In this repo:

```text
C = 4
H = W = 32
patch_size = 2
P = 256
d = 16
```

For other domains, adapt patchification:

```text
video:      patches over [T, H, W]
audio:      patches over [F, T]
sequence:   contiguous token blocks
3D volume:  local cubes
```

The required contract is:

```python
patchify(residual) -> [B, P, d]
unpatchify(patches) -> residual
```

Both directions should be exact before RAMA projection is added.

## 5. Add RAMA Projection

RAMA uses frozen orthogonal bases per patch:

```text
bases: [P, d, d]
```

Projection:

```text
y_p = patch_p A_p
```

Inverse:

```text
patch_p = y_p A_p^T
```

This gives a rotated coordinate system for each local residual patch. The current repo implements this in `src/rama/projector.py`.

Porting rules:

```text
Keep bases frozen.
Cache bases to disk.
Regenerate bases if P or d changes.
Test projection -> inverse roundtrip.
```

## 6. Choose Continuous or Discrete Micro Modeling

The current default is the discrete categorical route:

```text
RAMA coordinates y
  -> uniform scalar quantization
  -> integer tokens
  -> categorical logits
  -> cross-entropy loss
```

Default shapes:

```text
y:      [B, P, d]
tokens: [B, P, d]
logits: [B, P, d, V]
V:      256
```

This route is simple, robust, and easy to evaluate with reconstruction tests.

The optional continuous route models each RAMA coordinate with a conditional density, such as a spline flow. Use it when continuous likelihood quality matters more than implementation simplicity.

## 7. Build the Context Encoder

The context encoder maps macro latents to one conditioning vector per residual patch:

```text
context = E(z_L)
context: [B, P, D]
```

In the current repo:

```text
P = 256
D = 256
```

Context encoder options:

```text
small CNN
tiny ViT
DiT block stack
cross-attention encoder
MLP for non-spatial features
temporal transformer for video
```

The only hard requirement is alignment: context position `p` should condition residual patch `p`.

## 8. Train the Micro Model

The categorical micro training loop is:

```text
z
  -> z_L, z_H
  -> context = E(z_L)
  -> patches = patchify(z_H)
  -> y = RAMAProjector.project(patches)
  -> tokens = RAMATokenizer.quantize(y)
  -> logits = MicroModel(context)
  -> loss = cross_entropy(logits, tokens)
```

This trains:

```text
p(z_H | z_L)
```

The micro model should not need the full image, original pixels, or the decoder. It only needs cached latents and the decomposition rule.

## 9. Recombine and Decode

After sampling:

```text
z_L = sample_macro()
context = E(z_L)
tokens = sample_micro_tokens(context)
y_hat = dequantize(tokens)
patches_hat = inverse_rama(y_hat)
z_H_hat = unpatchify(patches_hat)
z_hat = upsample(z_L) + z_H_hat
output = decoder(z_hat)
```

If the host architecture is not an autoencoder, replace `decoder(z_hat)` with the downstream consumer of the reconstructed representation.

## 10. Minimal Porting Checklist

When moving this idea to another repo, implement these pieces in order:

```text
1. Latent adapter
   encode/decode or representation read/write.

2. Decomposition adapter
   z -> z_L, z_H and z_L + z_H -> z.

3. Patch adapter
   residual -> patches and patches -> residual.

4. RAMA bases
   create, cache, load, and verify [P, d, d] bases.

5. Projector
   project and inverse-project patches.

6. Tokenizer
   estimate bound, quantize, dequantize.

7. Macro model
   train and sample p(z_L).

8. Context encoder
   map z_L to [B, P, D].

9. Micro model
   train p(tokens | context) or p(y | context).

10. End-to-end sampler
    sample macro, sample residual, reconstruct full representation.
```

## 11. Things That Usually Change

Expect to change:

```text
latent shape
downsample factor
patch size
number of patches
patch dimension
context encoder architecture
macro generator architecture
tokenizer bound
number of token bins
sampler
decoder or output adapter
```

Try not to change these unless needed:

```text
macro/residual factorization
patchify/unpatchify roundtrip requirement
frozen orthogonal RAMA bases
project/inverse roundtrip requirement
quantization reconstruction test
real-z_L micro reconstruction test
full generated-z_L sampling test
```

