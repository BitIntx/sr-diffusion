from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets.manifest import crop_square, pil_to_tensor
from sr_diffusion.degradations import DegradationPipeline
from sr_diffusion.models import AutoencoderKL, ConditionalUNet, LRToLatentPredictor, NoiseScheduler
from sr_diffusion.utils import autocast_context, get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 conditional diffusion SR sampling.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-lr", type=Path, help="Low-resolution RGB input image.")
    input_group.add_argument("--input-hr", type=Path, help="HR image to center-crop and degrade for controlled eval.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", default="photo")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--init", choices=("noise", "condition"), default="condition")
    parser.add_argument("--start-timestep", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resize-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=0, help="Save intermediate samples every N sampler steps.")
    return parser.parse_args()


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    array = np.round(array * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def load_autoencoder(config: dict, device: torch.device) -> AutoencoderKL:
    auto_cfg = config["autoencoder"]
    vae_config = load_config(auto_cfg["config"])
    vae = AutoencoderKL.from_config(vae_config["model"]).to(device)
    checkpoint = torch.load(auto_cfg["checkpoint"], map_location=device)
    vae.load_state_dict(checkpoint["model"])
    vae.eval()
    return vae


def load_condition_encoder(config: dict, device: torch.device) -> LRToLatentPredictor:
    cond_cfg = config["condition_encoder"]
    cond_config = load_config(cond_cfg["config"])
    encoder = LRToLatentPredictor.from_config(cond_config["model"]).to(device)
    checkpoint = torch.load(cond_cfg["checkpoint"], map_location=device)
    encoder.load_state_dict(checkpoint["model"])
    encoder.eval()
    return encoder


def load_unet(config: dict, checkpoint_path: Path, device: torch.device) -> tuple[ConditionalUNet, int]:
    model = ConditionalUNet.from_config(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint.get("step", 0))


def prepare_inputs(args: argparse.Namespace, config: dict) -> tuple[Image.Image, Image.Image | None]:
    data_config = config["data"]
    hr_size = int(data_config["hr_size"])
    scale = int(data_config.get("scale", 4))
    lr_size = hr_size // scale
    rng = random.Random(args.seed)

    if args.input_hr:
        hr = Image.open(args.input_hr).convert("RGB")
        hr = crop_square(hr, hr_size, rng=rng, random_crop=False)
        pipeline = DegradationPipeline.from_preset(data_config.get("degradation_preset", "mild"), scale=scale)
        lr = pipeline.apply(hr, rng=rng, out_size=lr_size)
        return lr, hr

    lr = Image.open(args.input_lr).convert("RGB")
    if args.resize_lr:
        lr = crop_square(lr, lr_size, rng=rng, random_crop=False)
    elif lr.size != (lr_size, lr_size):
        raise ValueError(f"Expected LR size {(lr_size, lr_size)}, got {lr.size}. Use --resize-lr to resize/crop.")
    return lr, None


def make_timesteps(
    num_train_timesteps: int,
    num_steps: int,
    device: torch.device,
    start_timestep: int | None = None,
) -> torch.Tensor:
    steps = max(1, min(int(num_steps), int(num_train_timesteps)))
    start = num_train_timesteps - 1 if start_timestep is None else max(0, min(int(start_timestep), num_train_timesteps - 1))
    timesteps = torch.linspace(start, 0, steps, device=device).round().long()
    return torch.unique_consecutive(timesteps)


def resolve_start_timestep(config: dict, requested: int | None) -> int | None:
    if requested is not None:
        return requested
    value = config.get("sampling", {}).get("start_timestep")
    return None if value is None else int(value)


def ddim_sample(
    model: ConditionalUNet,
    vae: AutoencoderKL,
    condition_encoder: LRToLatentPredictor,
    scheduler: NoiseScheduler,
    lr: torch.Tensor,
    domain_id: torch.Tensor,
    steps: int,
    eta: float,
    init: str,
    start_timestep: int | None,
    dtype_name: str,
    seed: int,
    output_dir: Path,
    save_every: int = 0,
) -> torch.Tensor:
    device = lr.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if start_timestep is None and init == "condition":
        start_timestep = min(int(scheduler.num_train_timesteps) - 1, 50)
    timesteps = make_timesteps(scheduler.num_train_timesteps, steps, device=device, start_timestep=start_timestep)
    lr_input = lr.mul(2.0).sub(1.0)
    with torch.no_grad(), autocast_context(device, dtype_name):
        condition = condition_encoder(lr_input, domain_id)

    noise = torch.randn(
        condition.shape[0],
        model.latent_channels,
        condition.shape[-2],
        condition.shape[-1],
        device=device,
        generator=generator,
    )
    if init == "noise":
        latent = noise
    elif init == "condition":
        first_timestep = torch.full((condition.shape[0],), int(timesteps[0].item()), device=device, dtype=torch.long)
        latent = scheduler.add_noise(condition, noise, first_timestep)
    else:
        raise ValueError(f"Unsupported init: {init}")
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=latent.dtype)

    with torch.no_grad():
        for index, timestep in enumerate(timesteps):
            timestep_batch = torch.full((latent.shape[0],), int(timestep.item()), device=device, dtype=torch.long)
            with autocast_context(device, dtype_name):
                predicted_noise = model(latent, timestep_batch, condition, domain_id)
            predicted_noise = predicted_noise.to(dtype=latent.dtype)
            pred_x0 = scheduler.predict_x0_from_noise(latent, timestep_batch, predicted_noise)

            prev_timestep = int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
            alpha_t = alphas_cumprod[int(timestep.item())]
            alpha_prev = torch.ones((), device=device, dtype=latent.dtype) if prev_timestep < 0 else alphas_cumprod[prev_timestep]

            if prev_timestep < 0:
                latent = pred_x0
            else:
                sigma = eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t)) * torch.sqrt(
                    torch.clamp(1.0 - alpha_t / alpha_prev, min=0.0)
                )
                direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma.pow(2), min=0.0))
                latent = torch.sqrt(alpha_prev) * pred_x0 + direction_scale * predicted_noise
                if eta > 0.0:
                    latent = latent + sigma * torch.randn(latent.shape, device=device, dtype=latent.dtype, generator=generator)

            if save_every > 0 and ((index + 1) % save_every == 0 or index + 1 == len(timesteps)):
                with autocast_context(device, dtype_name):
                    preview = vae.decode(latent)
                for sample_idx, image in enumerate(denormalize(preview)):
                    tensor_to_pil(image).save(output_dir / f"sample_{sample_idx:02d}_step_{index + 1:03d}.png")

    with torch.no_grad(), autocast_context(device, dtype_name):
        decoded = vae.decode(latent)
    return denormalize(decoded)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device)
    dtype_name = config["train"].get("dtype", "bf16")
    data_config = config["data"]
    domains = data_config.get("domains", {"photo": 0, "anime": 1})
    if args.domain not in domains:
        raise ValueError(f"Unknown domain '{args.domain}'. Available: {sorted(domains)}")

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lr_image, gt_image = prepare_inputs(args, config)
    lr_image.save(args.output_dir / "input_lr.png")
    if gt_image is not None:
        gt_image.save(args.output_dir / "gt_hr.png")

    vae = load_autoencoder(config, device)
    condition_encoder = load_condition_encoder(config, device)
    model, checkpoint_step = load_unet(config, args.checkpoint, device)
    scheduler = NoiseScheduler.from_config(config.get("diffusion", {}))
    start_timestep = resolve_start_timestep(config, args.start_timestep)

    lr_tensor = pil_to_tensor(lr_image).unsqueeze(0).repeat(args.num_samples, 1, 1, 1).to(device)
    domain_id = torch.full((args.num_samples,), int(domains[args.domain]), device=device, dtype=torch.long)
    print(
        f"checkpoint_step={checkpoint_step} lr_size={lr_image.size} steps={args.steps} "
        f"eta={args.eta} init={args.init} start_timestep={start_timestep} "
        f"samples={args.num_samples} device={device}"
    )

    output = ddim_sample(
        model=model,
        vae=vae,
        condition_encoder=condition_encoder,
        scheduler=scheduler,
        lr=lr_tensor,
        domain_id=domain_id,
        steps=args.steps,
        eta=args.eta,
        init=args.init,
        start_timestep=start_timestep,
        dtype_name=dtype_name,
        seed=args.seed,
        output_dir=args.output_dir,
        save_every=args.save_every,
    )
    for index, image in enumerate(output):
        tensor_to_pil(image).save(args.output_dir / f"sr_{index:02d}.png")
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
