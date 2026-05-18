from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _norm(channels: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=max(1, math.gcd(channels, groups)), num_channels=channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = _norm(channels, groups)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _norm(channels, groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class LRToLatentPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        num_blocks: int = 8,
        norm_groups: int = 32,
        num_domains: int = 2,
    ) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.domain_embedding = nn.Embedding(num_domains, base_channels)
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels, groups=norm_groups) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            _norm(base_channels, norm_groups),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LRToLatentPredictor":
        return cls(
            in_channels=config.get("in_channels", 3),
            latent_channels=config.get("latent_channels", 16),
            base_channels=config.get("base_channels", 128),
            num_blocks=config.get("num_blocks", 8),
            norm_groups=config.get("norm_groups", 32),
            num_domains=config.get("num_domains", 2),
        )

    def forward(self, lr: torch.Tensor, domain_id: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input(lr)
        if domain_id is not None:
            domain_bias = self.domain_embedding(domain_id).unsqueeze(-1).unsqueeze(-1)
            x = x + domain_bias
        x = self.blocks(x)
        return self.output(x)
