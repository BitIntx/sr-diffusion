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

We finished the first **Stage 1: VAE / Autoencoder** pass, the first
**Stage 2: deterministic LR -> HR latent pretraining** pass, and the first
**Stage 3: conditional latent diffusion** pass. The current best sampled SR
checkpoint is the Stage 4 condition-start checkpoint initialized from Stage 3.
A conservative Stage 4-lite low-timestep fine-tune improved one-step diagnostics
but did not improve the fixed 32-step sampled validation result, so it is
recorded as an experiment rather than promoted as the best model.

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
- Stage 3 DDIM/img2img inference and sampled validation eval.
- Stage 4-lite low-timestep and condition-start fine-tuning.

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

Next scale-up Stage 2 config:

```text
configs/latent_pretrain_photo100k.yaml
```

This run uses batch size `64` on MI300X and `max_steps: 30000`, which is about
18.6 passes over the 103,450-image training split.

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

Next scale-up Stage 3 config:

```text
configs/diffusion_photo100k_b32.yaml
```

Current Stage 3 model:

```text
conditional U-Net params: 76.6M
frozen Stage 2 condition encoder params: 2.4M
latent shape: 16 x 128 x 128
batch size: 32
max steps: 25000
```

Selected Stage 3 checkpoint:

```text
/home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
```

Stage 3 training result:

```text
finished step: 25000
best eval noise/x0: step 24000, eval/noise_mse 0.00766, eval/x0_mse 0.09063
best decoded PSNR diagnostic: step 25000, eval/decoded_psnr 24.10
```

Sampled Stage 3 eval, using `--init condition`, `--start-timestep 50`,
and 32 DDIM steps on 32 fixed validation images:

```text
mean bicubic PSNR: 24.66
mean SR PSNR:      25.55
mean delta:        +0.89 dB
```

Sampled Stage 3 eval on all 100 validation images:

```text
mean bicubic PSNR: 24.478
mean SR PSNR:      25.222
mean delta:        +0.744 dB
```

Stage 4-lite low-timestep fine-tune result:

```text
config: configs/diffusion_photo10k_b32_stage4_lowt.yaml
initialized from: Stage 3 best checkpoint
train timesteps: 0..100
finished step: 5000
best eval/x0_mse: step 5000, eval/x0_mse 0.01186
best decoded PSNR diagnostic: step 4500, eval/decoded_psnr 32.74
sampled val32 SR PSNR: 25.5493
sampled val32 delta vs Stage 3: -0.0037 dB
decision: do not promote; keep Stage 3 as current best sampled checkpoint
```

Stage 4 condition-start fine-tune result:

```text
config: configs/diffusion_photo10k_b32_stage4_condition.yaml
selected checkpoint: /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt
initialized from: Stage 3 best checkpoint
train timesteps: 25..100
stopped early: step 2500, best checkpoint at step 1000
best one-step condition diagnostic: step 1000, eval/decoded_psnr 23.78
best sampled setting: --init condition --start-timestep 25 --steps 32
sampled val32 SR PSNR: 25.660
sampled val100 SR PSNR: 25.293
sampled val100 delta vs Stage 3: +0.071 dB
decision: promote as current best sampled checkpoint
```

This trains the low-timestep path from the Stage 2 condition latent instead of
from a noised ground-truth latent, matching the current inference initialization
more closely.

At `batch_size=32`, one epoch is:

```text
10000 images / 32 = 312.5 steps
```

So the Stage 3 `25000` step config is about `80` epochs.

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

The active scale-up target is:

```text
/home/jwheojjang/scratch/sr-diffusion/data/manifest_photo100k.csv
```

It is built from DF2K plus 100,000 deterministic COCO train2017 images selected
with short side `>=320`, for about 103,550 training images and 100 validation
images. COCO only has 45,897 train2017 images with short side `>=480`, so the
stricter high-resolution-only variant is closer to photo50k.

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

To recover the larger photo100k setup after scratch loss:

```bash
bash scripts/recover_scratch.sh --coco-count 100000
```

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

Run the photo100k Stage 2 scale-up from the selected 10k checkpoint:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_latent_pretrain.py \
  --config configs/latent_pretrain_photo100k.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/checkpoints/best_eval_latent.pt
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

After Stage 2 photo100k finishes, run the photo100k Stage 3 config:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_diffusion.py \
  --config configs/diffusion_photo100k_b32.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
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

- Done for the first 10k pass; photo100k scale-up is the next active pass.
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

Run the photo100k Stage 2 scale-up:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_latent_pretrain.py \
  --config configs/latent_pretrain_photo100k.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/checkpoints/best_eval_latent.pt
```

Stage 3: conditional latent diffusion

- First pass complete. It is the baseline for the current Stage 4 condition
  checkpoint.
- Train diffusion U-Net over HR latents.
- Conditioning:
  - frozen Stage 2 LR-to-latent condition encoder
  - timestep embedding
  - photo/anime domain embedding
- Initial model size is 76.6M trainable U-Net parameters.
- Target model size is roughly 250M-500M parameters after the pipeline is stable.

Stage 4: perceptual / GAN fine-tune

- Current stage.
- First conservative Stage 4-lite low-timestep fine-tune is complete. It
  improved one-step diagnostics but not the fixed 32-step sampled eval, so it
  is not promoted over Stage 3.
- Condition-start fine-tuning, initialized from the Stage 3 best checkpoint,
  is the current best sampled SR checkpoint. It trains low timesteps `25..100`,
  but starts the training noisy latent from the Stage 2 condition latent so the
  train path better matches `infer_diffusion.py --init condition`.
- It uses a small effective-noise loss plus a stronger x0 latent reconstruction
  loss to preserve fidelity. The best sampled setting so far is
  `--start-timestep 25`.
- Use carefully, because later perceptual/GAN tuning can improve apparent
  sharpness while hurting fidelity.

Run the Stage 4-lite low-timestep fine-tune:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_diffusion.py \
  --config configs/diffusion_photo10k_b32_stage4_lowt.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
```

Run the Stage 4 condition-start fine-tune:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_diffusion.py \
  --config configs/diffusion_photo10k_b32_stage4_condition.yaml \
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
eval_diffusion_samples.py sampled diffusion validation eval
compare_eval_samples.py   sampled eval comparison contact sheets
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

Run current best Stage 4 condition-start sampling from an HR image by creating
a controlled LR input first:

```bash
/home/jwheojjang/venvs/rocm/bin/python infer_diffusion.py \
  --config configs/diffusion_photo10k_b32_stage4_condition.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt \
  --input-hr /path/to/hr_image.png \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/infer_diffusion_stage4_condition \
  --steps 32 \
  --seed 123
```

The Stage 4 condition config sets `sampling.start_timestep: 25`, so the command
above uses the best sampled setting found so far unless `--start-timestep` is
passed explicitly.

Run Stage 3 baseline sampling from an HR image by creating a controlled LR input first:

```bash
/home/jwheojjang/venvs/rocm/bin/python infer_diffusion.py \
  --config configs/diffusion_photo10k_b32.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt \
  --input-hr /path/to/hr_image.png \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/infer_diffusion_sample \
  --steps 32 \
  --seed 123
```

Run Stage 3 baseline sampling from an existing LR image:

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
added (`--init condition`). If a config has `sampling.start_timestep`, that
value is used when `--start-timestep` is omitted. Otherwise condition sampling
falls back to `50`. Pure noise sampling is available with `--init noise`, but
the current checkpoints are more stable in condition-initialized mode.

Run a small sampled validation sweep and compare against bicubic:

```bash
/home/jwheojjang/venvs/rocm/bin/python eval_diffusion_samples.py \
  --config configs/diffusion_photo10k_b32_stage4_condition.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_stage4_condition_val8_32step \
  --split val \
  --limit 8 \
  --steps 32 \
  --seed 1337
```

The sampled eval grid is written as `grid_lr_bicubic_sr_gt.png`, with columns in
this order: LR nearest, bicubic, SR, GT.

Compare two sampled eval directories and create top win/loss contact sheets:

```bash
/home/jwheojjang/venvs/rocm/bin/python compare_eval_samples.py \
  --baseline-dir /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_stage3_val100_t50_32step \
  --candidate-dir /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_stage4_condition_val100_t25_32step \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/compare_stage3_vs_stage4_condition_val100 \
  --baseline-label stage3 \
  --candidate-label stage4cond
```

Reconstruct one image:

```bash
/home/jwheojjang/venvs/rocm/bin/python infer_reconstruct.py \
  --config configs/autoencoder_tiny.yaml \
  --checkpoint runs/autoencoder_tiny/checkpoints/latest.pt \
  --input runs/toy_data/images/0000.png \
  --output-dir runs/reconstruct_smoke
```
