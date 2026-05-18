from __future__ import annotations

import torch

from sr_diffusion.models import LRToLatentPredictor


def test_lr_to_latent_predictor_shape() -> None:
    model = LRToLatentPredictor(
        in_channels=3,
        latent_channels=8,
        base_channels=32,
        num_blocks=2,
        norm_groups=8,
        num_domains=2,
    )
    lr = torch.randn(2, 3, 32, 32)
    domain_id = torch.tensor([0, 1])
    latent = model(lr, domain_id)
    assert latent.shape == (2, 8, 32, 32)
