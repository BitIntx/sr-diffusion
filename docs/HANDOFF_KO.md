# VM Handoff / 대화 기반 인수인계

이 문서는 원문 채팅 로그가 아니라, 현재 대화에서 결정하고 실행한 내용을
다른 VM의 Codex/작업자가 바로 이어받을 수 있게 정리한 공개용 요약입니다.

## 목표

직접 학습하는 x4 vision-only latent diffusion super-resolution 모델.

- T2I pretrained diffusion 모델을 사용하지 않음.
- `LR 128x128 -> HR 512x512`가 현재 기본 목표.
- `LR 192x192 -> HR 768x768`은 이후 목표.
- photo / anime domain conditioning을 한 코드베이스에서 처리.
- PyTorch first.
- GPU/ROCm 우선, TPU/XLA는 나중에 고려.
- custom CUDA/ROCm op 없이 표준 PyTorch 위주로 유지.

## 모델 구조

```text
HR image
  -> factor-4 VAE / AutoencoderKL
  -> HR latent

LR image
  -> LR-to-latent condition encoder
  -> condition latent

noisy HR latent + condition latent + timestep + domain embedding
  -> conditional diffusion U-Net
  -> denoised HR latent
  -> VAE decoder
  -> x4 SR output
```

현재 파라미터 수:

```text
Stage 1 VAE:                  21.096M
Stage 2 LR-to-latent encoder:  2.388M
Stage 3 diffusion U-Net:      76.610M
Full inference path:         100.094M
```

## 라이선스 / 공개 상태

- GitHub: <https://github.com/BitIntx/sr-diffusion>
- Hugging Face: `jwheo/sr-diffusion`
- GitHub repo는 public.
- HF model repo도 public.
- code license: PolyForm Noncommercial 1.0.0.
- checkpoint/artifact license: CC BY-NC 4.0.
- 상업적 이용은 금지.
- raw training data는 repo/HF에 올리지 않음.

## 현재 구현된 것

- 프로젝트 scaffold.
- config system.
- manifest 기반 dataset loader.
- x4 degradation pipeline.
- Stage 1 VAE training/eval/inference.
- Stage 2 deterministic LR-to-HR-latent pretrain.
- Stage 3 conditional latent diffusion training.
- Stage 4 condition-start fine-tune prototype.
- W&B online logging.
- fixed sample image logging: LR / GT / Pred.
- sampled validation/eval tooling.
- HF artifact upload/download scripts.
- Colab demo.
- tiled inference:
  - `--tile`
  - 128x128 LR tiles
  - overlap feather blending
  - arbitrary-size LR image to x4 output

## 데이터 상태

Scratch root:

```text
/home/jwheojjang/scratch/sr-diffusion
```

현재 주요 manifests:

```text
data/manifest_photo10k.csv
data/manifest_photo100k.csv
```

`manifest_photo100k.csv`:

```text
photo/train: 103450
photo/val:   100
```

구성:

- DIV2K
- Flickr2K
- deterministic COCO train2017 subset

COCO train2017은 `min_size>=480` 조건으로는 45,897장밖에 안 나와서,
photo100k는 `min_size>=320` 기준으로 구성했다.

Scratch는 VM restart나 VM 이동 시 날아갈 수 있으므로 데이터는 복구
스크립트로 다시 받는 전제로 운영한다.

## 완료된 학습

### Stage 1: VAE / Autoencoder

```text
config: configs/autoencoder_photo10k.yaml
run: autoencoder_photo10k_b16_eval_online
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints/best_eval_recon.pt
finished step: 50000
eval/recon: 0.01198
eval/kl:    9.38684
eval/psnr:  40.19
```

HF:

```text
checkpoints/stage1_autoencoder_best_eval_recon.pt
```

### Stage 2: 10k LR-to-latent

```text
config: configs/latent_pretrain_photo10k.yaml
run: latent_pretrain_photo10k_b16
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/checkpoints/best_eval_latent.pt
finished step: 50000
best eval/latent_loss: step 48000, 0.21775
best decoded PSNR proxy: step 47000, 23.89
```

HF:

```text
checkpoints/stage2_latent_pretrain_best_eval_latent.pt
```

### Stage 3: 10k diffusion

```text
config: configs/diffusion_photo10k_b32.yaml
run: diffusion_photo10k_b32
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
finished step: 25000
best eval/noise_mse checkpoint: step 24000
sampled val100 t50 32-step:
  SR PSNR:      25.2216
  bicubic PSNR: 24.4778
  delta:        +0.7438
```

HF:

```text
checkpoints/stage3_diffusion_b32_best_eval_noise.pt
```

### Stage 4: 10k condition-start prototype

```text
config: configs/diffusion_photo10k_b32_stage4_condition.yaml
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt
best checkpoint step: 1000
val100 t25 32-step sampled SR PSNR: 25.2930
```

HF:

```text
checkpoints/stage4_condition_b32_best_eval_condition_decoded.pt
metrics/stage4_condition_val100_t25_32step_summary.json
```

### Stage 2: photo100k scale-up

```text
config: configs/latent_pretrain_photo100k.yaml
run: latent_pretrain_photo100k_b64
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_b64/checkpoints/best_eval_latent.pt
finished step: 30000
best eval/latent_loss: step 28000, 0.21230
best decoded PSNR proxy: step 22000, 23.93
final eval: step 30000, eval/latent_loss 0.21267, decoded_psnr 23.88
```

HF:

```text
checkpoints/stage2_photo100k_b64_best_eval_latent.pt
metrics/stage2_photo100k_b64_summary.json
```

### Stage 3: photo100k scale-up

```text
config: configs/diffusion_photo100k_b32.yaml
run: diffusion_photo100k_b32
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32/checkpoints/best_eval_noise.pt
initialized from:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
condition encoder:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_b64/checkpoints/best_eval_latent.pt
finished step: 60000
best eval/noise_mse: step 60000, 0.00680
best decoded PSNR proxy: step 53000, 24.57
final eval: step 60000, decoded_psnr 24.56, eval/x0_mse 0.08052
```

HF:

```text
checkpoints/stage3_photo100k_b32_best_eval_noise.pt
metrics/stage3_photo100k_b32_summary.json
```

## 현재 관찰 / 판단

- 기본 x4 복원력은 잡혔다.
- 디노이즈와 선명화 능력은 아직 약하다.
- 원인은 현재 degradation이 `mild`이고 Stage3가 noise prediction baseline에
  가깝기 때문.
- 다음 개선축은 데이터/모델 크기보다 degradation v2와 Stage4 loss 설계다.

## 다음 작업

우선순위:

1. Stage3 photo100k checkpoint의 sampled eval 실행.
2. photo100k 기반 Stage4 condition-start fine-tune.
3. degradation v2 설계:
   - stronger blur
   - Gaussian/sensor noise 강화
   - JPEG/WebP artifact 강화
   - ringing
   - oversharpen artifact
   - color shift / banding
4. perceptual/fidelity fine-tune:
   - `x0_weight`를 켠 condition-start training
   - LPIPS/VGG perceptual loss 검토
   - GAN은 나중에, A/B eval 기반으로 조심스럽게
5. few-step distillation.
6. A/B Elo preference eval.

## 새 VM에서 Codex에게 줄 짧은 프롬프트

```text
이 repo는 /home/.../sr-diffusion 의 x4 latent diffusion SR 프로젝트다.
docs/HANDOFF_KO.md 와 docs/VM_RECOVERY_KO.md 를 먼저 읽고 이어서 작업해줘.
현재 Stage3 photo100k까지 완료됐고, 다음은 sampled eval 후 photo100k Stage4
condition-start fine-tune 및 degradation v2/denoise-sharpening 개선이다.
상업적 이용은 금지이고, raw dataset은 GitHub/HF에 올리지 않는다.
```
