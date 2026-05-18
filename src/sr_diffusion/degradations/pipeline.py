from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_degradation_preset(
    name: str,
    preset_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(preset_path) if preset_path else _repo_root() / "configs" / "degradation_presets.yaml"
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    presets = data.get("presets", {})
    if name not in presets:
        available = ", ".join(sorted(presets))
        raise KeyError(f"Unknown degradation preset '{name}'. Available: {available}")
    return presets[name]


def _chance(rng: random.Random, probability: float) -> bool:
    return rng.random() < float(probability)


def _range_value(rng: random.Random, value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if len(value) != 2:
        raise ValueError(f"Expected scalar or [min, max], got {value}")
    return rng.uniform(float(value[0]), float(value[1]))


def _range_int(rng: random.Random, value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if len(value) != 2:
        raise ValueError(f"Expected int or [min, max], got {value}")
    return rng.randint(int(value[0]), int(value[1]))


def _to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0, 255).round().astype(np.uint8)


class DegradationPipeline:
    def __init__(self, config: dict[str, Any], scale: int = 4):
        self.config = config
        self.scale = scale

    @classmethod
    def from_preset(cls, name: str, scale: int = 4) -> "DegradationPipeline":
        return cls(load_degradation_preset(name), scale=scale)

    def apply(
        self,
        image: Image.Image,
        rng: random.Random | None = None,
        out_size: int | None = None,
    ) -> Image.Image:
        rng = rng or random.Random()
        cfg = self.config
        hr = image.convert("RGB")

        if _chance(rng, cfg.get("blur_prob", 0.0)):
            radius = _range_value(rng, cfg.get("blur_radius"), 0.6)
            hr = hr.filter(ImageFilter.GaussianBlur(radius=radius))

        target_size = out_size or max(1, min(hr.size) // self.scale)
        resample = rng.choice(
            [
                Image.Resampling.BICUBIC,
                Image.Resampling.BILINEAR,
                Image.Resampling.LANCZOS,
            ]
        )
        lr = hr.resize((target_size, target_size), resample=resample)

        if _chance(rng, cfg.get("gaussian_noise_prob", 0.0)):
            sigma = _range_value(rng, cfg.get("gaussian_sigma"), 2.0)
            lr = self._gaussian_noise(lr, sigma=sigma, rng=rng)

        if _chance(rng, cfg.get("poisson_noise_prob", 0.0)):
            lr = self._poisson_noise(lr, rng=rng)

        if _chance(rng, cfg.get("jpeg_prob", 0.0)):
            quality = _range_int(rng, cfg.get("jpeg_quality"), 90)
            lr = self._compress(lr, fmt="JPEG", quality=quality)

        if _chance(rng, cfg.get("webp_prob", 0.0)):
            quality = _range_int(rng, cfg.get("webp_quality"), 90)
            lr = self._compress(lr, fmt="WEBP", quality=quality)

        if _chance(rng, cfg.get("color_prob", 0.0)):
            jitter = cfg.get("color_jitter", [0.95, 1.05])
            lr = ImageEnhance.Color(lr).enhance(_range_value(rng, jitter, 1.0))
            lr = ImageEnhance.Contrast(lr).enhance(_range_value(rng, jitter, 1.0))
            lr = ImageEnhance.Brightness(lr).enhance(_range_value(rng, jitter, 1.0))

        if _chance(rng, cfg.get("anime_line_prob", 0.0)):
            lr = self._anime_line_change(lr, rng=rng)

        if _chance(rng, cfg.get("banding_prob", 0.0)):
            levels = _range_int(rng, cfg.get("banding_levels"), 64)
            lr = self._banding(lr, levels=levels)

        if _chance(rng, cfg.get("sharpen_prob", 0.0)):
            factor = _range_value(rng, cfg.get("sharpen_factor"), 1.2)
            lr = ImageEnhance.Sharpness(lr).enhance(factor)

        return lr.convert("RGB")

    @staticmethod
    def _gaussian_noise(image: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
        array = np.asarray(image).astype(np.float32)
        np_rng = np.random.default_rng(rng.randrange(2**32))
        array = array + np_rng.normal(0.0, sigma, size=array.shape)
        return Image.fromarray(_to_uint8(array), mode="RGB")

    @staticmethod
    def _poisson_noise(image: Image.Image, rng: random.Random) -> Image.Image:
        array = np.asarray(image).astype(np.float32) / 255.0
        np_rng = np.random.default_rng(rng.randrange(2**32))
        peak = max(8.0, 2.0 ** math.ceil(math.log2(255.0)))
        noised = np_rng.poisson(array * peak) / peak
        return Image.fromarray(_to_uint8(noised * 255.0), mode="RGB")

    @staticmethod
    def _compress(image: Image.Image, fmt: str, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        try:
            image.save(buffer, format=fmt, quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        except OSError:
            return image

    @staticmethod
    def _anime_line_change(image: Image.Image, rng: random.Random) -> Image.Image:
        if rng.random() < 0.5:
            filtered = image.filter(ImageFilter.MinFilter(3))
        else:
            filtered = image.filter(ImageFilter.MaxFilter(3))
        return Image.blend(image, filtered, alpha=rng.uniform(0.08, 0.22))

    @staticmethod
    def _banding(image: Image.Image, levels: int) -> Image.Image:
        levels = max(2, int(levels))
        array = np.asarray(image).astype(np.float32)
        step = 255.0 / float(levels - 1)
        array = np.round(array / step) * step
        return Image.fromarray(_to_uint8(array), mode="RGB")
