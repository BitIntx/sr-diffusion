from __future__ import annotations

import torch
from torch import nn

from sr_diffusion.utils import load_matching_weights


def test_load_matching_weights_skips_shape_mismatches() -> None:
    target = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    source = {
        "0.weight": torch.ones_like(target[0].weight),
        "0.bias": torch.ones_like(target[0].bias),
        "1.weight": torch.ones(3, 8),
        "missing.weight": torch.ones(1),
    }

    stats = load_matching_weights(target, source)

    assert stats["matched_tensors"] == 2
    assert stats["skipped_shape"] == 1
    assert stats["skipped_missing"] == 1
    assert torch.all(target[0].weight == 1)
    assert torch.all(target[0].bias == 1)
