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
