from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _norm(channels: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=max(1, math.gcd(channels, groups)), num_channels=channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = _norm(in_channels, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _norm(out_channels, groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


@dataclass
class AutoencoderOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 96,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4),
        latent_channels: int = 16,
        num_res_blocks: int = 2,
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if len(channel_multipliers) < 3:
            raise ValueError("channel_multipliers must contain at least 3 entries for factor-4 latents")

        channels = [base_channels * int(mult) for mult in channel_multipliers]
        self.latent_channels = latent_channels

        encoder: list[nn.Module] = [nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)]
        current = channels[0]
        for level, out_ch in enumerate(channels):
            for _ in range(num_res_blocks):
                encoder.append(ResBlock(current, out_ch, groups=norm_groups))
                current = out_ch
            if level < 2:
                next_ch = channels[level + 1]
                encoder.append(Downsample(current, next_ch))
                current = next_ch
        encoder.extend(
            [
                ResBlock(current, current, groups=norm_groups),
                _norm(current, norm_groups),
                nn.SiLU(),
                nn.Conv2d(current, latent_channels * 2, kernel_size=3, padding=1),
            ]
        )
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = [
            nn.Conv2d(latent_channels, current, kernel_size=3, padding=1),
            ResBlock(current, current, groups=norm_groups),
        ]
        for level in reversed(range(len(channels))):
            out_ch = channels[level]
            for _ in range(num_res_blocks):
                decoder.append(ResBlock(current, out_ch, groups=norm_groups))
                current = out_ch
            if level <= 2 and level > 0:
                next_ch = channels[level - 1]
                decoder.append(Upsample(current, next_ch))
                current = next_ch
        decoder.extend(
            [
                _norm(current, norm_groups),
                nn.SiLU(),
                nn.Conv2d(current, out_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            ]
        )
        self.decoder = nn.Sequential(*decoder)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoencoderKL":
        return cls(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            base_channels=config.get("base_channels", 96),
            channel_multipliers=config.get("channel_multipliers", [1, 2, 4]),
            latent_channels=config.get("latent_channels", 16),
            num_res_blocks=config.get("num_res_blocks", 2),
            norm_groups=config.get("norm_groups", 32),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moments = self.encoder(x)
        mean, logvar = torch.chunk(moments, chunks=2, dim=1)
        return mean, logvar.clamp(min=-30.0, max=20.0)

    @staticmethod
    def sample(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> AutoencoderOutput:
        mean, logvar = self.encode(x)
        latent = self.sample(mean, logvar) if sample_posterior else mean
        reconstruction = self.decode(latent)
        return AutoencoderOutput(
            reconstruction=reconstruction,
            latent=latent,
            mean=mean,
            logvar=logvar,
        )
