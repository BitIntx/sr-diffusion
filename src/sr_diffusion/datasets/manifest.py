from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from sr_diffusion.degradations import DegradationPipeline


@dataclass(frozen=True)
class ManifestEntry:
    path: Path
    domain: str
    split: str


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def crop_square(image: Image.Image, size: int, rng: random.Random, random_crop: bool) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    if min(width, height) < size:
        scale = size / float(min(width, height))
        new_size = (max(size, round(width * scale)), max(size, round(height * scale)))
        image = image.resize(new_size, resample=Image.Resampling.LANCZOS)
        width, height = image.size

    max_x = width - size
    max_y = height - size
    if random_crop and (max_x > 0 or max_y > 0):
        left = rng.randint(0, max_x) if max_x > 0 else 0
        top = rng.randint(0, max_y) if max_y > 0 else 0
    else:
        left = max_x // 2
        top = max_y // 2
    return image.crop((left, top, left + size, top + size))


class ManifestImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        hr_size: int,
        scale: int,
        domains: dict[str, int],
        degradation_preset: str = "mild",
        seed: int = 0,
        deterministic: bool | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.hr_size = int(hr_size)
        self.scale = int(scale)
        self.lr_size = self.hr_size // self.scale
        self.domains = domains
        self.seed = int(seed)
        self.deterministic = split != "train" if deterministic is None else deterministic
        self.random_crop = split == "train"
        self.entries = self._load_entries()
        self.degradation_preset = degradation_preset
        self.default_pipeline = DegradationPipeline.from_preset(degradation_preset, scale=scale)
        self.anime_pipeline = DegradationPipeline.from_preset("anime", scale=scale)

        if self.hr_size % self.scale != 0:
            raise ValueError(f"hr_size must be divisible by scale: {self.hr_size}, {self.scale}")
        if not self.entries:
            raise ValueError(f"No entries for split '{split}' in {self.manifest_path}")

    @classmethod
    def from_config(cls, data_config: dict[str, Any], seed: int = 0) -> "ManifestImageDataset":
        return cls(
            manifest_path=data_config["manifest"],
            split=data_config.get("split", "train"),
            hr_size=data_config.get("hr_size", 512),
            scale=data_config.get("scale", 4),
            domains=data_config.get("domains", {"photo": 0, "anime": 1}),
            degradation_preset=data_config.get("degradation_preset", "mild"),
            seed=seed,
        )

    def _load_entries(self) -> list[ManifestEntry]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        base_dir = self.manifest_path.parent
        entries: list[ManifestEntry] = []
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"path", "domain", "split"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Manifest must contain columns: {sorted(required)}")
            for row in reader:
                if row["split"] != self.split:
                    continue
                image_path = Path(row["path"])
                if not image_path.is_absolute():
                    image_path = base_dir / image_path
                domain = row["domain"]
                if domain not in self.domains:
                    raise ValueError(f"Unknown domain '{domain}' for {image_path}")
                entries.append(ManifestEntry(path=image_path, domain=domain, split=row["split"]))
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        rng = random.Random(self.seed + index) if self.deterministic else random
        image = Image.open(entry.path).convert("RGB")
        hr = crop_square(image, self.hr_size, rng=rng, random_crop=self.random_crop)
        pipeline = self.anime_pipeline if self.degradation_preset == "domain" and entry.domain == "anime" else self.default_pipeline
        lr = pipeline.apply(hr, rng=rng, out_size=self.lr_size)

        return {
            "hr": pil_to_tensor(hr),
            "lr": pil_to_tensor(lr),
            "domain_id": torch.tensor(self.domains[entry.domain], dtype=torch.long),
            "domain": entry.domain,
            "path": str(entry.path),
        }
