# sr-diffusion

[![Technical Report](https://img.shields.io/badge/technical_report-PDF-blue)](paper/sr_diffusion_report.pdf)
[![Code License](https://img.shields.io/badge/code_license-PolyForm_Noncommercial_1.0.0-orange)](LICENSE)
[![Checkpoint License](https://img.shields.io/badge/checkpoints-CC_BY--NC_4.0-orange)](CHECKPOINT_LICENSE.md)

Vision-only x4 latent diffusion super-resolution experiments.
Report source: [paper/main.tex](paper/main.tex).

This is a public source-available, non-commercial research project. The goal is
to train an SR model directly, without using a pretrained text-to-image
diffusion model. The intended final model handles photo and anime/illustration
domains in one codebase with domain conditioning.

This repository is not OSI-approved open source because commercial use is not
permitted.

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

We finished the first **Stage 1: VAE / Autoencoder** pass and the first
**Stage 2: deterministic LR -> HR latent pretraining** pass. We are now moving
into **Stage 3: conditional latent diffusion**.

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
- Stage 3 conditional U-Net, noise scheduler, and diffusion training loop.

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

Selected Stage 2 checkpoint:

```text
/home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/checkpoints/best_eval_latent.pt
```

Stage 2 final result:

```text
finished step: 50000
best eval latent loss: step 48000, eval/latent_loss 0.21775
best decoded PSNR proxy: step 47000, eval/decoded_psnr 23.89
```

Current Stage 3 config:

```text
configs/diffusion_photo10k_b32.yaml
```

Current Stage 3 model:

```text
conditional U-Net params: 76.6M
frozen Stage 2 condition encoder params: 2.4M
latent shape: 16 x 128 x 128
batch size: 32
max steps: 25000
```

Current sampled Stage 3 eval, using `--init condition`, `--start-timestep 50`,
and 32 DDIM steps on 32 fixed validation images:

```text
mean bicubic PSNR: 24.66
mean SR PSNR:      25.55
mean delta:        +0.89 dB
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

## Hugging Face

Hugging Face is used as persistent checkpoint storage because scratch can be
lost after VM restarts. The current target is a public model repository:

```text
jwheo/sr-diffusion
```

Upload only selected checkpoints/configs/metrics, not raw datasets. See
[docs/HUGGINGFACE.md](docs/HUGGINGFACE.md) for the exact upload commands.

## License

Code is released under the [PolyForm Noncommercial License 1.0.0](LICENSE).
Model checkpoints and generated artifacts are released under
[CC BY-NC 4.0](CHECKPOINT_LICENSE.md).

Commercial use is not permitted without separate written permission. This
includes paid hosted inference, resale, or integration into commercial
products.

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

Run the current Stage 3 conditional diffusion config:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_diffusion.py \
  --config configs/diffusion_photo10k_b32.yaml
```

Recommended Stage 3 tmux launch:

```bash
tmux new-session -d -s sr_stage3 \
  'cd /home/jwheojjang/sr-diffusion && env PYTHONUNBUFFERED=1 /home/jwheojjang/venvs/rocm/bin/python train_diffusion.py --config configs/diffusion_photo10k_b32.yaml > /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/train_tmux.log 2>&1'
```

Watch the Stage 3 log:

```bash
tail -f /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/train_tmux.log
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

- Done for the first pass.
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

- Current stage.
- Train diffusion U-Net over HR latents.
- Conditioning:
  - frozen Stage 2 LR-to-latent condition encoder
  - timestep embedding
  - photo/anime domain embedding
- Initial model size is 76.6M trainable U-Net parameters.
- Target model size is roughly 250M-500M parameters after the pipeline is stable.

Stage 4: perceptual / GAN fine-tune

- Current next step is a conservative Stage 4-lite low-timestep diffusion
  fine-tune before adding perceptual/GAN losses.
- Initialize from the Stage 3 best checkpoint.
- Train only timesteps `0..100`, matching the sampled SR path where
  `--start-timestep 50` worked best.
- Add a small x0 latent reconstruction loss to preserve fidelity while
  sharpening the diffusion correction.
- Use carefully, because later perceptual/GAN tuning can improve apparent
  sharpness while hurting fidelity.

Run the Stage 4-lite low-timestep fine-tune:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_diffusion.py \
  --config configs/diffusion_photo10k_b32_stage4_lowt.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
```

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
  models/                 AutoencoderKL, LR predictor, diffusion U-Net
train_autoencoder.py      Stage 1 training entrypoint
train_latent_pretrain.py  Stage 2 deterministic latent pretraining entrypoint
train_diffusion.py        Stage 3 conditional diffusion training entrypoint
infer_diffusion.py        Stage 3 DDIM/img2img SR sampling entrypoint
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

Run a tiny Stage 3 smoke test:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_diffusion.py \
  --config configs/diffusion_scratch_tiny.yaml \
  --limit-steps 1
```

Run Stage 3 sampling from an HR image by creating a controlled LR input first:

```bash
/home/jwheojjang/venvs/rocm/bin/python infer_diffusion.py \
  --config configs/diffusion_photo10k_b32.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt \
  --input-hr /path/to/hr_image.png \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/infer_diffusion_sample \
  --steps 32 \
  --seed 123
```

Run Stage 3 sampling from an existing LR image:

```bash
/home/jwheojjang/venvs/rocm/bin/python infer_diffusion.py \
  --config configs/diffusion_photo10k_b32.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt \
  --input-lr /path/to/lr_128.png \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/infer_diffusion_sample \
  --steps 32 \
  --seed 123
```

The default sampler starts from the Stage 2 condition latent with light noise
added (`--init condition`, `--start-timestep 50`). Pure noise sampling is
available with `--init noise`, but the current Stage 3 checkpoint is more stable
in condition-initialized mode.

Run a small sampled validation sweep and compare against bicubic:

```bash
/home/jwheojjang/venvs/rocm/bin/python eval_diffusion_samples.py \
  --config configs/diffusion_photo10k_b32.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_b32_val8_32step \
  --split val \
  --limit 8 \
  --steps 32 \
  --seed 1337
```

The sampled eval grid is written as `grid_lr_bicubic_sr_gt.png`, with columns in
this order: LR nearest, bicubic, SR, GT.

Reconstruct one image:

```bash
/home/jwheojjang/venvs/rocm/bin/python infer_reconstruct.py \
  --config configs/autoencoder_tiny.yaml \
  --checkpoint runs/autoencoder_tiny/checkpoints/latest.pt \
  --input runs/toy_data/images/0000.png \
  --output-dir runs/reconstruct_smoke
```
