from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image

from sr_diffusion.eval import evaluate_autoencoder, make_eval_loader
from sr_diffusion.models import AutoencoderKL


def test_autoencoder_eval_returns_metrics(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (96, 96), (64, 128, 192)).save(images / "a.png")
    Image.new("RGB", (96, 96), (192, 128, 64)).save(images / "b.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        writer.writerow({"path": "images/a.png", "domain": "photo", "split": "val"})
        writer.writerow({"path": "images/b.png", "domain": "photo", "split": "val"})

    config = {
        "data": {
            "manifest": str(manifest),
            "hr_size": 64,
            "scale": 4,
            "degradation_preset": "clean",
            "domains": {"photo": 0, "anime": 1},
            "num_workers": 0,
        },
        "loss": {
            "reconstruction": "charbonnier",
            "kl_weight": 1e-6,
        },
    }
    model = AutoencoderKL(
        base_channels=16,
        channel_multipliers=[1, 2, 4],
        latent_channels=4,
        num_res_blocks=1,
        norm_groups=4,
    )
    dataloader = make_eval_loader(config, split="val", seed=0, batch_size=2, limit=2, num_workers=0)
    metrics = evaluate_autoencoder(model, dataloader, torch.device("cpu"), "fp32", config["loss"])

    assert metrics["eval/num_images"] == 2.0
    assert metrics["eval/recon"] > 0.0
    assert metrics["eval/psnr"] > 0.0
