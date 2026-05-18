from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.losses import vae_loss
from sr_diffusion.utils import autocast_context


def make_eval_dataset(
    config: dict[str, Any],
    split: str,
    seed: int,
    limit: int | None = None,
) -> ManifestImageDataset | Subset:
    data_config = config["data"]
    dataset = ManifestImageDataset(
        manifest_path=data_config["manifest"],
        split=split,
        hr_size=data_config.get("hr_size", 512),
        scale=data_config.get("scale", 4),
        domains=data_config.get("domains", {"photo": 0, "anime": 1}),
        degradation_preset=data_config.get("degradation_preset", "mild"),
        seed=seed,
        deterministic=True,
    )
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def make_eval_loader(
    config: dict[str, Any],
    split: str,
    seed: int,
    batch_size: int,
    limit: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    dataset = make_eval_dataset(config, split=split, seed=seed, limit=limit)
    workers = int(config["data"].get("num_workers", 0) if num_workers is None else num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def evaluate_autoencoder(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
    loss_config: dict[str, Any],
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "kl": 0.0,
        "mse": 0.0,
    }
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            hr = batch["hr"].to(device, non_blocking=True)
            target = hr.mul(2.0).sub(1.0)
            batch_size = int(target.shape[0])

            with autocast_context(device, dtype_name):
                output = model(target, sample_posterior=False)
                _, metrics = vae_loss(
                    output.reconstruction,
                    target,
                    output.mean,
                    output.logvar,
                    loss_config,
                )
                mse = torch.nn.functional.mse_loss(output.reconstruction, target)

            totals["loss"] += metrics["loss"] * batch_size
            totals["recon"] += metrics["recon"] * batch_size
            totals["kl"] += metrics["kl"] * batch_size
            totals["mse"] += float(mse.detach().cpu()) * batch_size
            count += batch_size

    if was_training:
        model.train()

    count = max(1, count)
    mse_value = totals["mse"] / count
    # Target/reconstruction tensors are in [-1, 1], so the peak-to-peak range is 2.
    psnr = 20.0 * math.log10(2.0) - 10.0 * math.log10(max(mse_value, 1e-12))
    return {
        "eval/loss": totals["loss"] / count,
        "eval/recon": totals["recon"] / count,
        "eval/kl": totals["kl"] / count,
        "eval/mse": mse_value,
        "eval/psnr": psnr,
        "eval/num_images": float(count),
    }


def save_eval_metrics(path: Path, step: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
