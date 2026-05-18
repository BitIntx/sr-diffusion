# sr-diffusion

Vision-only x4 latent diffusion super-resolution experiments.

The first milestone is Stage 0-1:

- Project scaffold and config loading.
- Manifest-based photo/anime dataset.
- Deterministic x4 degradation presets.
- Factor-4 AutoencoderKL for 512 -> 128 latents.
- Autoencoder training and reconstruction smoke test.

This repo avoids custom CUDA ops. ROCm devices are accessed through PyTorch's
standard `cuda` device abstraction.

## Quick smoke test

Create a toy dataset:

```bash
python scripts/make_toy_dataset.py --output runs/toy_data --count 16
```

## Scratch disk

This VM exposes an ext4 scratch partition labeled `DOSCRATCH`. Mount it before
large datasets or long training runs:

```bash
bash scripts/mount_doscratch.sh
```

The default mount point is `/home/jwheojjang/scratch`. Pass another mount point
as the first argument if needed. If the disk was mounted read-only, run the same
script again; it will try to remount it read-write.

The scratch volume is treated as ephemeral. After a VM restart, recover the
scratch layout and development datasets with:

```bash
bash scripts/recover_scratch.sh
```

That recreates directories, the toy dataset, DIV2K, Flickr2K, and the combined
DF2K manifest. Add `--smoke` if you also want a 1-step training check after
recovery.

To recover only the smaller DIV2K seed dataset, skip Flickr2K:

```bash
bash scripts/recover_scratch.sh --skip-flickr2k
```

The default recovery creates
`/home/jwheojjang/scratch/sr-diffusion/data/manifest_df2k_photo.csv` for
[configs/autoencoder_df2k.yaml](configs/autoencoder_df2k.yaml).

Expected real dataset layout:

```text
/home/jwheojjang/scratch/sr-diffusion/datasets/photo/...
/home/jwheojjang/scratch/sr-diffusion/datasets/anime/...
```

Build a manifest:

```bash
python scripts/build_manifest.py \
  --photo-dir /home/jwheojjang/scratch/sr-diffusion/datasets/photo \
  --anime-dir /home/jwheojjang/scratch/sr-diffusion/datasets/anime \
  --output /home/jwheojjang/scratch/sr-diffusion/data/manifest.csv
```

Download DIV2K HR images for the first photo-domain VAE run:

```bash
python scripts/download_div2k.py
python scripts/dataset_report.py \
  --manifest /home/jwheojjang/scratch/sr-diffusion/data/manifest_div2k_photo.csv
```

See [docs/DATASETS.md](docs/DATASETS.md) for dataset notes and licensing caveats.

W&B logging is enabled in scratch configs in offline mode. Sync a run later with
the `wandb sync ...` command printed at the end of training, or switch
`logging.wandb.mode` to `online` after `wandb login`.

Run tests:

```bash
pytest
```

Train a tiny autoencoder for a few steps:

```bash
python train_autoencoder.py --config configs/autoencoder_tiny.yaml --limit-steps 10
```

Reconstruct one image:

```bash
python infer_reconstruct.py \
  --config configs/autoencoder_tiny.yaml \
  --checkpoint runs/autoencoder_tiny/checkpoints/latest.pt \
  --input runs/toy_data/images/0000.png \
  --output-dir runs/reconstruct_smoke
```
