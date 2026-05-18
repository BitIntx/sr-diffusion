# sr-diffusion

[![Technical Report](https://img.shields.io/badge/technical_report-PDF-blue)](paper/sr_diffusion_report.pdf)

Vision-only x4 latent diffusion super-resolution experiments.
Report source: [paper/main.tex](paper/main.tex).

This project is for study/research. The goal is to train an SR model directly,
without using a pretrained text-to-image diffusion model. The intended final
model handles photo and anime/illustration domains in one codebase with domain
conditioning.

## Goal

Target task:

```text
LR 128x128 -> HR 512x512
LR 192x192 -> HR 768x768 later
```

Planned model:

```text
HR image
  -> factor-4 VAE / autoencoder
  -> HR latent

LR image
  -> condition encoder
  -> multi-scale LR features

noisy HR latent + LR features + timestep + domain embedding
  -> conditional diffusion U-Net
  -> denoised HR latent
  -> VAE decoder
  -> x4 SR output
```

Constraints:

- PyTorch first.
- ROCm/GPU primary.
- No custom CUDA/ROCm ops.
- TPU/XLA compatibility is a later consideration, so code should stay close to
  standard PyTorch where practical.
- No pretrained T2I model dependency.

## Current Status

We finished the first **Stage 1: VAE / Autoencoder** pass and are now starting
**Stage 2: deterministic LR -> HR latent pretraining**.

Implemented:

- Project scaffold and config loading.
- Manifest-based dataset loader with `photo` / `anime` domain IDs.
- On-the-fly x4 degradation pipeline.
- Factor-4 `AutoencoderKL`.
- Autoencoder training loop with bf16 autocast.
- W&B online/offline logging.
- Fixed validation sample logging for Stage 1:
  - `samples/LR`
  - `samples/GT`
  - `samples/HR`
- Validation eval during training:
  - `eval/loss`
  - `eval/recon`
  - `eval/kl`
  - `eval/mse`
  - `eval/psnr`
  - `eval/num_images`
- Standalone checkpoint eval script.
- Scratch recovery scripts for ephemeral VM storage.
- Stage 2 LR-to-latent predictor and training loop.

Stage 1 training config:

```text
configs/autoencoder_photo10k.yaml
```

Stage 1 run name:

```text
autoencoder_photo10k_b16_eval_online
```

Selected Stage 1 VAE checkpoint:

```text
/home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints/best_eval_recon.pt
```

Stage 1 VAE shape:

```text
HR 512x512 -> latent 128x128
latent channels: 16
batch size: 16
max steps: 100000
train set: 10000 photo images
val set: 100 photo images
eval: every 1000 steps
fixed sample logging: every 500 steps
```

The first Stage 1 pass was stopped at step `50000`. The selected checkpoint is
`best_eval_recon.pt`, which matched the 50k checkpoint in the current run:

```text
eval/recon: 0.01198
eval/kl:    9.38684
eval/psnr:  40.19
```

Current Stage 2 config:

```text
configs/latent_pretrain_photo10k.yaml
```

Current Stage 2 run name:

```text
latent_pretrain_photo10k_b16
```

At `batch_size=16`, one epoch is:

```text
10000 images / 16 = 625 steps
```

So the current `100000` step config is about `160` epochs.

## Data

The current photo training manifest is:

```text
/home/jwheojjang/scratch/sr-diffusion/data/manifest_photo10k.csv
```

It contains:

```text
photo/train: 10000
photo/val:   100
```

The 10k photo set is built from:

- DIV2K HR.
- Flickr2K HR.
- A deterministic subset of COCO train2017.

LR images are not stored. They are generated on the fly from HR crops by the
degradation pipeline. The current `mild` degradation already includes light LR
noise:

- Gaussian noise with probability `0.25`, sigma `[0.0, 4.0]`.
- Poisson noise with probability `0.05`.
- JPEG/WebP compression.
- blur, color jitter, sharpening, and mild banding.

For VAE training, LR is only used for visual logging. The VAE loss is:

```text
HR -> encode -> latent -> decode -> reconstructed HR
```

LR degradation becomes a core training signal in Stage 2/3.

See [docs/DATASETS.md](docs/DATASETS.md) for dataset notes and licensing caveats.

## Scratch Disk

This VM exposes an ext4 scratch partition labeled `DOSCRATCH`. Mount it before
large datasets or long training runs:

```bash
bash scripts/mount_doscratch.sh
```

The default mount point is:

```text
/home/jwheojjang/scratch
```

The scratch volume is treated as ephemeral. After a VM restart, recover the
scratch layout and development datasets with:

```bash
bash scripts/recover_scratch.sh
```

That recreates:

- scratch directories
- toy dataset
- DIV2K
- Flickr2K
- COCO train2017 subset
- `manifest_photo10k.csv`

To recover only the smaller DIV2K seed dataset:

```bash
bash scripts/recover_scratch.sh --skip-flickr2k --skip-coco
```

## Training

Run the current Stage 1 VAE training config:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_autoencoder.py \
  --config configs/autoencoder_photo10k.yaml
```

Recommended long-running launch through tmux:

```bash
tmux new-session -d -s sr_ae10k \
  'cd /home/jwheojjang/sr-diffusion && env PYTHONUNBUFFERED=1 /home/jwheojjang/venvs/rocm/bin/python train_autoencoder.py --config configs/autoencoder_photo10k.yaml > /home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/train_tmux.log 2>&1'
```

Watch the training log:

```bash
tail -f /home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/train_tmux.log
```

Run the current Stage 2 deterministic latent pretraining config:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_latent_pretrain.py \
  --config configs/latent_pretrain_photo10k.yaml
```

Recommended Stage 2 tmux launch:

```bash
tmux new-session -d -s sr_stage2 \
  'cd /home/jwheojjang/sr-diffusion && env PYTHONUNBUFFERED=1 /home/jwheojjang/venvs/rocm/bin/python train_latent_pretrain.py --config configs/latent_pretrain_photo10k.yaml > /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/train_tmux.log 2>&1'
```

Watch the Stage 2 log:

```bash
tail -f /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/train_tmux.log
```

Watch GPU usage:

```bash
watch -n 1 rocm-smi --showuse --showmemuse --showtemp --showpower
```

Attach to the tmux session:

```bash
tmux attach -t sr_ae10k
```

Detach without stopping training:

```text
Ctrl-b d
```

## Eval

Training eval is enabled in `configs/autoencoder_photo10k.yaml`:

```yaml
eval:
  enabled: true
  split: val
  limit: 100
  batch_size: 16
  every: 1000
  run_at_start: true
```

This means:

- eval at step `1`
- eval at step `1000`
- eval at step `2000`
- and so on

The best checkpoint by `eval/recon` is written to:

```text
checkpoints/best_eval_recon.pt
```

Manual checkpoint eval:

```bash
/home/jwheojjang/venvs/rocm/bin/python eval_autoencoder.py \
  --config configs/autoencoder_photo10k.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints/latest.pt \
  --split val \
  --limit 100
```

## W&B

The current config logs to W&B online:

```yaml
logging:
  wandb:
    project: sr-diffusion
    name: autoencoder_photo10k_b16_eval_online
    mode: online
```

Image logging uses fixed validation images so improvements are comparable over
time:

```yaml
logging:
  samples:
    split: val
    count: 4
    indices: [0, 1, 2, 3]
```

Logged image keys:

- `samples/LR`: degraded LR, upsampled for viewing.
- `samples/GT`: original HR target.
- `samples/HR`: VAE reconstruction.

The name `samples/HR` currently means reconstructed HR output. If this becomes
confusing, rename it to `samples/Recon` before the next large run.

## Project Roadmap

Stage 0: scaffold and data pipeline

- Done.
- Repo scaffold, configs, manifests, degradation pipeline, smoke tests.

Stage 1: VAE / Autoencoder

- Done for the first pass.
- Train factor-4 VAE on 512 HR crops.
- Select checkpoint using fixed visual samples plus `eval/recon`, `eval/psnr`,
  and residual qualitative checks.
- Possible improvements before moving on:
  - LPIPS/perceptual eval.
  - perceptual training loss.
  - KL weight sweep.
  - larger or domain-balanced data.
  - rename `samples/HR` to `samples/Recon` for clarity.

Stage 2: deterministic LR -> HR latent pretrain

- Current stage.
- Freeze the selected Stage 1 VAE.
- Train an LR-to-latent predictor that maps degraded LR inputs to HR VAE
  encoder means.
- This is where LR degradation quality starts to matter directly.
- Log fixed validation `samples/LR`, `samples/GT`, and `samples/Pred` to W&B.

Run the current Stage 2 pretraining config:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_latent_pretrain.py \
  --config configs/latent_pretrain_photo10k.yaml
```

Stage 3: conditional latent diffusion

- Train diffusion U-Net over HR latents.
- Conditioning:
  - multi-scale LR features
  - timestep embedding
  - photo/anime domain embedding
- Target model size is roughly 250M-500M parameters after the pipeline is stable.

Stage 4: perceptual / GAN fine-tune

- Later-stage quality tuning after diffusion works.
- Use carefully, because perceptual/GAN tuning can improve apparent sharpness
  while hurting fidelity.

Stage 5: few-step distillation

- Distill the diffusion model for faster inference.

Stage 6: preference eval

- Fixed private eval set.
- Generate outputs from multiple checkpoints/settings.
- A/B comparisons.
- Accumulate Elo separately for photo and anime.

## Repo Layout

```text
configs/                  experiment configs
docs/                     dataset and project notes
scripts/                  dataset, scratch, and utility scripts
src/sr_diffusion/         package code
  datasets/               manifest dataset
  degradations/           x4 LR degradation pipeline
  eval/                   eval helpers
  losses/                 reconstruction/KL losses
  models/                 AutoencoderKL and future models
train_autoencoder.py      Stage 1 training entrypoint
train_latent_pretrain.py  Stage 2 deterministic latent pretraining entrypoint
eval_autoencoder.py       standalone VAE eval entrypoint
infer_reconstruct.py      reconstruction smoke/inference
tests/                    unit tests
```

## Smoke Tests

Create a toy dataset:

```bash
/home/jwheojjang/venvs/rocm/bin/python scripts/make_toy_dataset.py \
  --output runs/toy_data \
  --count 16
```

Train a tiny autoencoder for a few steps:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_autoencoder.py \
  --config configs/autoencoder_tiny.yaml \
  --limit-steps 10
```

Run unit tests:

```bash
/home/jwheojjang/venvs/rocm/bin/python -m pytest
```

Reconstruct one image:

```bash
/home/jwheojjang/venvs/rocm/bin/python infer_reconstruct.py \
  --config configs/autoencoder_tiny.yaml \
  --checkpoint runs/autoencoder_tiny/checkpoints/latest.pt \
  --input runs/toy_data/images/0000.png \
  --output-dir runs/reconstruct_smoke
```
