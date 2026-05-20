# Hugging Face Artifacts

Hugging Face is used as checkpoint storage for artifacts that should survive
scratch disk loss. The default target is a public model repository:

```text
jwheo/sr-diffusion
```

Keep dataset files and validation images out of the Hub repository unless their
licenses are reviewed. Upload configs, metrics, and selected checkpoints only.

The Hub repository uses `license: cc-by-nc-4.0` for checkpoints and artifacts.
The source code is separately licensed under PolyForm Noncommercial 1.0.0.

## Auth

Check the active login:

```bash
/home/jwheojjang/venvs/rocm/bin/python - <<'PY'
from huggingface_hub import whoami
print(whoami()["name"])
PY
```

## Download For Inference

From a fresh GitHub clone, install dependencies and download the selected public
prototype checkpoints:

```bash
python scripts/download_hf_checkpoints.py
```

This creates the local `checkpoints/` files expected by
`configs/hf/diffusion_stage4_condition.yaml`.

Run the default Stage 4 condition-start prototype:

```bash
python infer_diffusion.py \
  --input-lr /path/to/lr_128.png \
  --output-dir outputs/demo
```

Run the same checkpoint in tiled mode for larger LR images:

```bash
python infer_diffusion.py \
  --input-lr /path/to/larger_lr.png \
  --output-dir outputs/tiled_demo \
  --tile \
  --tile-overlap 32
```

The default `infer_diffusion.py` config is the HF-friendly Stage 4 config. It
uses relative checkpoint paths, so it works outside the original training VM.

## Upload Selected Artifacts

Upload the selected Stage 1 VAE and the current Stage 2 checkpoint:

```bash
/home/jwheojjang/venvs/rocm/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/sr-diffusion \
  --repo-type model \
  --update-card \
  --message "Upload Stage 1 and Stage 2 checkpoints" \
  --artifact LICENSE=LICENSE \
  --artifact CHECKPOINT_LICENSE.md=CHECKPOINT_LICENSE.md \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints/best_eval_recon.pt=checkpoints/stage1_autoencoder_best_eval_recon.pt \
  --artifact configs/autoencoder_photo10k.yaml=configs/autoencoder_photo10k.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/checkpoints/latest.pt=checkpoints/stage2_latent_pretrain_latest.pt \
  --artifact configs/latent_pretrain_photo10k.yaml=configs/latent_pretrain_photo10k.yaml
```

For a dry run:

```bash
/home/jwheojjang/venvs/rocm/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/sr-diffusion \
  --dry-run \
  --artifact configs/latent_pretrain_photo10k.yaml=configs/latent_pretrain_photo10k.yaml
```

Upload the current best sampled Stage 4 condition-start checkpoint and metrics:

```bash
/home/jwheojjang/venvs/rocm/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/sr-diffusion \
  --repo-type model \
  --message "Upload Stage 4 condition-start checkpoint" \
  --artifact configs/diffusion_photo10k_b32_stage4_condition.yaml=configs/diffusion_photo10k_b32_stage4_condition.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt=checkpoints/stage4_condition_b32_best_eval_condition_decoded.pt \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_stage4_condition_val100_t25_32step/summary.json=metrics/stage4_condition_val100_t25_32step_summary.json
```

Upload the current photo100k Stage 4 condition-start checkpoint and the v2
degradation config:

```bash
/home/jwheojjang/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/sr-diffusion \
  --repo-type model \
  --message "Upload photo100k Stage 4 and v2 configs" \
  --artifact configs/diffusion_photo100k_b32_stage4_condition.yaml=configs/diffusion_photo100k_b32_stage4_condition.yaml \
  --artifact configs/latent_pretrain_photo100k_v2.yaml=configs/latent_pretrain_photo100k_v2.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt=checkpoints/stage4_photo100k_condition_b32_best_eval_condition_decoded.pt \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_photo100k_stage4_condition_val100_t25_32step_final/summary.json=metrics/stage4_photo100k_condition_val100_t25_32step_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/compare_photo100k_stage3_vs_stage4_condition_val100/summary.json=metrics/stage4_photo100k_condition_compare_stage3_summary.json
```

Upload the photo100k Stage 2 `photo_v2` condition encoder and follow-up v2
diffusion configs:

```bash
/home/jwheojjang/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/sr-diffusion \
  --repo-type model \
  --message "Upload photo100k Stage 2 v2 condition encoder" \
  --artifact configs/diffusion_photo100k_b32_v2.yaml=configs/diffusion_photo100k_b32_v2.yaml \
  --artifact configs/diffusion_photo100k_b32_stage4_condition_v2.yaml=configs/diffusion_photo100k_b32_stage4_condition_v2.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/checkpoints/best_eval_latent.pt=checkpoints/stage2_photo100k_v2_b64_best_eval_latent.pt \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/summary.json=metrics/stage2_photo100k_v2_b64_summary.json
```

Make the Hub repository public after license files and the model card are in
place:

```bash
/home/jwheojjang/venvs/rocm/bin/python - <<'PY'
from huggingface_hub import HfApi
HfApi().update_repo_settings("jwheo/sr-diffusion", repo_type="model", private=False)
PY
```

## Policy

- Upload only checkpoints worth preserving.
- Prefer `best_eval_*.pt` over every intermediate step checkpoint.
- Use public visibility only with the non-commercial license files and model
  card in place.
- Do not upload raw training data or generated validation grids by default.
