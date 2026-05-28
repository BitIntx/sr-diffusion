from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).pow(2) + eps * eps).mean()


def _depthwise_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = int(image.shape[1])
    weight = kernel.to(device=image.device, dtype=image.dtype).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return F.conv2d(image, weight, padding=1, groups=channels)


def sobel_edge_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    sobel_x = prediction.new_tensor(
        [
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ]
    ) / 8.0
    sobel_y = prediction.new_tensor(
        [
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ]
    ) / 8.0
    pred_x = _depthwise_filter(prediction, sobel_x)
    pred_y = _depthwise_filter(prediction, sobel_y)
    target_x = _depthwise_filter(target, sobel_x)
    target_y = _depthwise_filter(target, sobel_y)
    return 0.5 * (charbonnier_loss(pred_x, target_x, eps=eps) + charbonnier_loss(pred_y, target_y, eps=eps))


def laplacian_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    kernel = prediction.new_tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    ) / 4.0
    pred_high = _depthwise_filter(prediction, kernel)
    target_high = _depthwise_filter(target, kernel)
    return charbonnier_loss(pred_high, target_high, eps=eps)


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
