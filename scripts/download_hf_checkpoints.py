from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_FILES = [
    "checkpoints/stage1_autoencoder_best_eval_recon.pt",
    "checkpoints/stage2_latent_pretrain_best_eval_latent.pt",
    "checkpoints/stage3_diffusion_b32_best_eval_noise.pt",
    "checkpoints/stage4_condition_b32_best_eval_condition_decoded.pt",
    "CHECKPOINT_LICENSE.md",
    "LICENSE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public sr-diffusion Hugging Face inference artifacts.")
    parser.add_argument("--repo-id", default="jwheo/sr-diffusion")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="Specific repo file to download. Can be repeated. Defaults to the prototype inference set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = args.file or DEFAULT_FILES
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        destination = args.output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        cached_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            filename=filename,
            local_dir=args.output_dir,
        )
        print(f"{filename} -> {cached_path}")


if __name__ == "__main__":
    main()
