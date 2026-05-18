# Datasets

The training pipeline uses HR images only. LR inputs are generated on the fly by
the degradation pipeline, so we do not store duplicated LR/HR pairs.

## Current Photo Seed

DIV2K HR is downloaded to:

```text
/home/jwheojjang/scratch/sr-diffusion/datasets/photo/div2k
```

Manifest:

```text
/home/jwheojjang/scratch/sr-diffusion/data/manifest_div2k_photo.csv
```

DIV2K is useful for early super-resolution development, but its page states it
is for academic research only. This project is currently for study/research, so
DIV2K is acceptable as the first development dataset.

The scratch disk can be wiped after VM restart. Recover the current DF2K setup
with:

```bash
bash scripts/recover_scratch.sh
```

## DF2K Expansion

DIV2K alone has 900 images. The current default photo-domain SR set adds
Flickr2K. Flickr2K is commonly used together with DIV2K in SR research and
contains 2,650 2K-resolution HR images. Recover or download it with:

```bash
bash scripts/recover_scratch.sh
```

This creates:

```text
/home/jwheojjang/scratch/sr-diffusion/data/manifest_flickr2k_photo.csv
/home/jwheojjang/scratch/sr-diffusion/data/manifest_df2k_photo.csv
```

The DF2K photo manifest should contain 3,550 rows total: 800 DIV2K train, 100
DIV2K validation, and 2,650 Flickr2K training images.

## Scaling Photo Data

Good next candidates:

- Unsplash Dataset Lite: 25k images, commercial and non-commercial usage stated
  by Unsplash.
- Open Images: very large and more operationally complex; useful after the VAE
  and degradation pipeline are stable.

## Anime / Illustration Data

Do not blindly pull scraped booru datasets for this project unless the intended
use and licensing constraints are clear. Many images are fan art or copyrighted
work. Prefer:

- User-owned or explicitly licensed illustrations.
- Public-domain / CC illustration collections with tracked attribution.
- Small private validation sets separated from training.

Expected layout for user-provided data:

```text
/home/jwheojjang/scratch/sr-diffusion/datasets/photo/...
/home/jwheojjang/scratch/sr-diffusion/datasets/anime/...
```

Build a combined manifest:

```bash
python scripts/build_manifest.py \
  --photo-dir /home/jwheojjang/scratch/sr-diffusion/datasets/photo \
  --anime-dir /home/jwheojjang/scratch/sr-diffusion/datasets/anime \
  --output /home/jwheojjang/scratch/sr-diffusion/data/manifest.csv
```
