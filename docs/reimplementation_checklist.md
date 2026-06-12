# Reimplementation Checklist

Use this checklist when implementing the Latent-RAMA pattern in another repository.

The goal is to avoid mixing architectural experiments with basic verification. Each stage should produce an artifact that can be inspected before moving on.

## Stage 0: Decide the Target

Define the new target architecture:

```text
Domain:
Dataset:
Representation / latent:
Decoder or output adapter:
Macro model:
Micro model:
Evaluation metric:
```

Write down the target shapes:

```text
full representation:
macro representation:
residual representation:
patch shape:
number of patches:
patch dimension:
context dimension:
token bins:
```

## Stage 1: Representation Adapter

Implement or identify:

```text
encode(input) -> z
decode(z) -> output
```

For a pretrained autoencoder, freeze it first. Do not train the autoencoder and RAMA system at the same time in the first pass.

Verification artifact:

```text
input vs reconstruction grid
mean reconstruction error or perceptual sanity check
saved latent tensor with expected shape
```

## Stage 2: Decomposition

Implement:

```text
decompose(z) -> z_L, z_H
reconstruct(z_L, z_H) -> z
```

For image latents, the default is:

```text
z_L = avg_pool2d(z, kernel_size=2, stride=2)
z_H = z - bilinear_upsample(z_L)
```

Verification artifact:

```text
max absolute error between z and reconstruct(decompose(z))
decoder output for:
  full z
  upsample(z_L)
  reconstruct(z_L, z_H)
```

## Stage 3: Latent Cache

Cache representations before training the generators.

Recommended cache item:

```python
{
    "z": full_latent,
    "z_L": macro_latent,      # optional but useful
    "z_H": residual_latent,   # optional but useful
    "path": source_id,
}
```

Verification artifact:

```text
number of cached items
example tensor shapes
dtype
min / max / mean / std
```

## Stage 4: Macro Generator

Train a model for:

```text
p(z_L)
```

Minimum required interface:

```python
train_macro(batch_z_l)
sample_macro(num_samples) -> z_L
```

Verification artifact:

```text
training loss curve
macro-only decoded samples
macro sampler smoke test
checkpoint with config
```

Do not start micro training until macro-only generation works at least as a rough baseline.

## Stage 5: Patchify Residuals

Implement:

```text
patchify(z_H) -> patches
unpatchify(patches) -> z_H
```

Expected shape:

```text
patches: [B, P, d]
```

Verification artifact:

```text
max absolute patchify roundtrip error
asserted shape contract
```

## Stage 6: RAMA Bases and Projector

Create frozen orthogonal bases:

```text
bases: [P, d, d]
```

Implement:

```text
project(patches) -> y
inverse(y) -> patches
```

Verification artifact:

```text
orthogonality error
projection inverse roundtrip error
cached bases file
```

Regenerate bases whenever `P` or `d` changes.

## Stage 7: Tokenizer

For the categorical route, estimate a quantization bound from real projected residuals:

```text
y = project(patchify(z_H))
bound = percentile(abs(y))
```

Then implement:

```text
quantize(y) -> tokens
dequantize(tokens) -> y_hat
```

Verification artifact:

```text
tokenizer config file
token min / max
quantization-only reconstruction grid
quantization error stats
```

If quantization-only reconstruction is poor, fix the tokenizer before training the micro model.

## Stage 8: Context Encoder

Implement:

```text
context_encoder(z_L) -> context
```

Expected shape:

```text
context: [B, P, D]
```

Verification artifact:

```text
context shape test
finite values test
alignment check between context positions and residual patches
```

## Stage 9: Micro Model

For the categorical route:

```text
input:  context [B, P, D]
output: logits  [B, P, d, V]
target: tokens  [B, P, d]
loss:   cross entropy
```

Verification artifact:

```text
single batch overfit test
training loss curve
real-z_L micro reconstruction grid
checkpoint with config
```

The real-z_L reconstruction test uses the ground-truth macro latent and only asks the micro model to reconstruct residual detail. This isolates micro quality from macro sampling quality.

## Stage 10: Full Sampler

Implement:

```text
z_L = sample_macro()
context = context_encoder(z_L)
tokens = sample_micro(context)
y_hat = dequantize(tokens)
patches_hat = inverse_rama(y_hat)
z_H_hat = unpatchify(patches_hat)
z_hat = reconstruct(z_L, z_H_hat)
output = decode(z_hat)
```

Verification artifact:

```text
macro-only samples
micro-only residual visualization, if meaningful
full generated samples
FID or task-specific metric
```

## Stage 11: Ablations

Useful ablations:

```text
macro only
macro + zero residual
real z_L + sampled z_H
generated z_L + sampled z_H
categorical micro vs continuous micro
different patch sizes
different token bin counts
different context encoders
different macro generators
```

## Common Failure Modes

Shape mismatch:

```text
P and d changed but cached RAMA bases were not regenerated.
```

Bad reconstructions before training:

```text
decomposition or patchify/unpatchify is not roundtripping.
```

Poor quantization-only reconstructions:

```text
tokenizer bound is too small, too large, or estimated on the wrong tensor.
```

Micro loss decreases but samples are bad:

```text
token sampling temperature is too high, context alignment is wrong, or real-z_L reconstruction was skipped.
```

Full samples are bad while real-z_L micro reconstructions are good:

```text
macro generator quality is the bottleneck.
```

Macro samples look good but full samples get noisy:

```text
micro model, tokenizer, or residual scale is the bottleneck.
```

