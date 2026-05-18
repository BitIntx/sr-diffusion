from __future__ import annotations

from typing import Any

import torch


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).pow(2) + eps * eps).mean()


def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).mean()


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    recon_kind = config.get("reconstruction", "charbonnier")
    if recon_kind == "l1":
        recon = torch.nn.functional.l1_loss(reconstruction, target)
    elif recon_kind == "mse":
        recon = torch.nn.functional.mse_loss(reconstruction, target)
    elif recon_kind == "charbonnier":
        recon = charbonnier_loss(reconstruction, target)
    else:
        raise ValueError(f"Unsupported reconstruction loss: {recon_kind}")

    kl = kl_loss(mean, logvar)
    kl_weight = float(config.get("kl_weight", 1e-6))
    total = recon + kl_weight * kl
    metrics = {
        "loss": float(total.detach().cpu()),
        "recon": float(recon.detach().cpu()),
        "kl": float(kl.detach().cpu()),
    }
    return total, metrics
