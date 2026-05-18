from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.losses import vae_loss
from sr_diffusion.models import AutoencoderKL
from sr_diffusion.utils import autocast_context, get_device, load_config, save_config, seed_everything, seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 1 autoencoder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def clean_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if not k.startswith("_")}


def init_wandb(config: dict[str, Any], output_dir: Path, model: torch.nn.Module) -> Any | None:
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging is enabled, but wandb is not installed") from exc

    wandb_dir = Path(wandb_cfg.get("dir", output_dir / "wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    mode = wandb_cfg.get("mode", "offline")
    os.environ.setdefault("WANDB_MODE", str(mode))

    run = wandb.init(
        project=wandb_cfg.get("project", "sr-diffusion"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name", config.get("project", {}).get("name")),
        dir=str(wandb_dir),
        mode=mode,
        tags=wandb_cfg.get("tags"),
        config=clean_config(config),
    )
    if bool(wandb_cfg.get("watch", False)):
        wandb.watch(model, log="gradients", log_freq=int(wandb_cfg.get("watch_log_freq", 100)))
    return run


def wandb_log(run: Any | None, data: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(data, step=step)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": clean_config(config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    seed_everything(seed)

    output_dir = Path(config["project"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    train_cfg = config["train"]
    device = get_device(train_cfg.get("device", "auto"))
    dtype_name = train_cfg.get("dtype", "bf16")
    print(f"device={device} dtype={dtype_name}")

    dataset = ManifestImageDataset.from_config(config["data"], seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )

    model = AutoencoderKL.from_config(config["model"]).to(device)
    if bool(train_cfg.get("compile", False)):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, device)
        print(f"resumed step={start_step}")

    max_steps = int(args.limit_steps or train_cfg.get("max_steps", 1000))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    sample_every = int(train_cfg.get("sample_every", 500))

    run = init_wandb(config, output_dir, model)
    if run is not None:
        wandb_log(
            run,
            {
                "dataset/num_images": len(dataset),
                "train/batch_size": int(train_cfg.get("batch_size", 1)),
                "train/grad_accum_steps": grad_accum_steps,
            },
            step=start_step,
        )

    model.train()
    step = start_step
    last_log = time.time()
    last_log_step = step
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps:
        for batch in dataloader:
            step += 1
            hr = batch["hr"].to(device, non_blocking=True)
            target = hr.mul(2.0).sub(1.0)

            with autocast_context(device, dtype_name):
                output = model(target, sample_posterior=True)
                loss, metrics = vae_loss(
                    output.reconstruction,
                    target,
                    output.mean,
                    output.logvar,
                    config["loss"],
                )
                scaled_loss = loss / grad_accum_steps

            scaled_loss.backward()

            if step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0 or step == 1:
                elapsed = max(1e-6, time.time() - last_log)
                interval_steps = max(1, step - last_log_step)
                last_log = time.time()
                last_log_step = step
                print(
                    f"step={step} loss={metrics['loss']:.5f} "
                    f"recon={metrics['recon']:.5f} kl={metrics['kl']:.5f} "
                    f"steps_per_sec={interval_steps / elapsed:.2f}"
                )
                wandb_log(
                    run,
                    {
                        "train/loss": metrics["loss"],
                        "train/recon": metrics["recon"],
                        "train/kl": metrics["kl"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "system/steps_per_sec": interval_steps / elapsed,
                    },
                    step=step,
                )

            if step % sample_every == 0 or step == 1:
                with torch.no_grad():
                    sample = torch.cat(
                        [
                            denormalize(target[:4]).float().cpu(),
                            denormalize(output.reconstruction[:4]).float().cpu(),
                        ],
                        dim=0,
                    )
                    sample_path = samples_dir / f"step_{step:07d}.png"
                    save_image(sample, sample_path, nrow=4)
                    if run is not None:
                        import wandb

                        wandb_log(run, {"samples/reconstruction_grid": wandb.Image(str(sample_path))}, step=step)

            if step % save_every == 0 or step == max_steps:
                save_checkpoint(checkpoints_dir / f"step_{step:07d}.pt", model, optimizer, step, config)
                save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, config)

            if step >= max_steps:
                break

    print(f"finished step={step}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
