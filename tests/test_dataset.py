from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from sr_diffusion.datasets import ManifestImageDataset


def test_manifest_dataset_returns_hr_lr(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (96, 96), (255, 0, 0)).save(images / "a.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        writer.writerow({"path": "images/a.png", "domain": "photo", "split": "train"})

    dataset = ManifestImageDataset(
        manifest_path=manifest,
        split="train",
        hr_size=64,
        scale=4,
        domains={"photo": 0, "anime": 1},
        degradation_preset="clean",
        seed=0,
    )
    item = dataset[0]
    assert item["hr"].shape == (3, 64, 64)
    assert item["lr"].shape == (3, 16, 16)
    assert int(item["domain_id"]) == 0
