# RAMA

Latent-RAMA experiments for SD-VAE latents and flow matching.

The first runnable milestone is a Stable Diffusion VAE sanity check:

1. Load a Hugging Face SD-VAE checkpoint.
2. Encode a 256x256 image normalized to `[-1, 1]`.
3. Decompose the latent into macro and residual components.
4. Decode VAE, macro-only, and full-decomposition reconstructions.

See [docs/sdvae.md](docs/sdvae.md) for the experiment notes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The preferred SD-VAE checkpoint is loaded from Hugging Face:

```text
stabilityai/sd-vae-ft-mse
```

This setup also supports a local materialized copy:

```bash
hf download stabilityai/sd-vae-ft-mse --local-dir checkpoints/sd-vae-ft-mse
```

If your Hugging Face environment requires authentication:

```bash
hf auth login
```

## VAE Sanity Check

Put a test image at `data/celeba256/example.jpg`, or pass another path:

```bash
python scripts/check_sdvae.py --image data/celeba256/example.jpg --checkpoint checkpoints/sd-vae-ft-mse --out outputs/sdvae_check
```

The script writes:

- `vae_reconstruction.png`
- `macro_reconstruction.png`
- `zh_reconstruction.png`
- `full_decomposition_reconstruction.png`
- `latents.pt`

## Macro Flow Training

Cache SD-VAE latents before training:

```bash
python scripts/cache_sdvae_latents.py --images data/celeba256 --out data/latents --checkpoint checkpoints/sd-vae-ft-mse --store-components
```

Start a single-process training run:

```bash
python scripts/train_macro_flow.py --config configs/celeba256_sdvae_macro.yaml
```

For multi-GPU training, launch through Accelerate:

```bash
accelerate launch scripts/train_macro_flow.py --config configs/celeba256_sdvae_macro.yaml
```

The trainer reads the `logging.tracker: wandb` block from the config and logs loss, velocity norms, gradient norm, and learning rate to Weights & Biases.
