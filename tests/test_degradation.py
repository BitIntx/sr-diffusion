from __future__ import annotations

import random

from PIL import Image

from sr_diffusion.degradations import DegradationPipeline


def test_degradation_outputs_expected_size() -> None:
    image = Image.new("RGB", (128, 128), (200, 120, 80))
    pipeline = DegradationPipeline.from_preset("mild", scale=4)
    lr = pipeline.apply(image, rng=random.Random(0), out_size=32)
    assert lr.size == (32, 32)
    assert lr.mode == "RGB"
