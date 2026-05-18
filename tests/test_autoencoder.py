from __future__ import annotations

import torch

from sr_diffusion.models import AutoencoderKL


def test_autoencoder_factor4_shapes() -> None:
    model = AutoencoderKL(
        base_channels=16,
        channel_multipliers=[1, 2, 4],
        latent_channels=8,
        num_res_blocks=1,
        norm_groups=8,
    )
    x = torch.randn(2, 3, 64, 64)
    output = model(x, sample_posterior=False)
    assert output.latent.shape == (2, 8, 16, 16)
    assert output.reconstruction.shape == x.shape
