from __future__ import annotations

import math
from typing import Any

import torch


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    gathered = values.to(device=timesteps.device).gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1.0 + s) * math.pi * 0.5).pow(2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999).float()


class NoiseScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.num_train_timesteps)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt((1.0 / alphas_cumprod) - 1.0)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NoiseScheduler":
        return cls(
            num_train_timesteps=config.get("num_train_timesteps", 1000),
            beta_schedule=config.get("beta_schedule", "linear"),
            beta_start=config.get("beta_start", 0.0001),
            beta_end=config.get("beta_end", 0.02),
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)

    def add_noise(self, original: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, original.shape).to(dtype=original.dtype)
        sqrt_one_minus_alpha = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, original.shape).to(
            dtype=original.dtype
        )
        return sqrt_alpha * original + sqrt_one_minus_alpha * noise

    def predict_x0_from_noise(self, noisy: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_recip = _extract(self.sqrt_recip_alphas_cumprod, timesteps, noisy.shape).to(dtype=noisy.dtype)
        sqrt_recipm1 = _extract(self.sqrt_recipm1_alphas_cumprod, timesteps, noisy.shape).to(dtype=noisy.dtype)
        return sqrt_recip * noisy - sqrt_recipm1 * noise

    def noise_from_x0(self, noisy: torch.Tensor, original: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, noisy.shape).to(dtype=noisy.dtype)
        sqrt_one_minus_alpha = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy.shape).to(
            dtype=noisy.dtype
        )
        return (noisy - sqrt_alpha * original) / sqrt_one_minus_alpha.clamp_min(1e-8)
