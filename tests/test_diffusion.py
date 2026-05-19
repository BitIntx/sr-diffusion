from __future__ import annotations

import torch

from sr_diffusion.models import ConditionalUNet, NoiseScheduler


def test_noise_scheduler_shapes() -> None:
    scheduler = NoiseScheduler(num_train_timesteps=10)
    x0 = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([0, 9])
    noisy = scheduler.add_noise(x0, noise, timesteps)
    pred_x0 = scheduler.predict_x0_from_noise(noisy, timesteps, noise)
    recovered_noise = scheduler.noise_from_x0(noisy, x0, timesteps)
    assert noisy.shape == x0.shape
    assert pred_x0.shape == x0.shape
    assert torch.isfinite(pred_x0).all()
    assert torch.allclose(recovered_noise, noise, atol=2e-4)


def test_conditional_unet_shape() -> None:
    model = ConditionalUNet(
        latent_channels=8,
        condition_channels=8,
        out_channels=8,
        base_channels=32,
        channel_multipliers=[1, 2],
        num_res_blocks=1,
        norm_groups=8,
        num_heads=4,
        attention_resolutions=[16],
        base_resolution=32,
        num_domains=2,
    )
    noisy = torch.randn(2, 8, 32, 32)
    condition = torch.randn(2, 8, 32, 32)
    timesteps = torch.tensor([1, 5])
    domain_id = torch.tensor([0, 1])
    prediction = model(noisy, timesteps, condition, domain_id)
    assert prediction.shape == noisy.shape
