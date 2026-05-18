from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small synthetic SR smoke-test dataset.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def make_photo_like(size: int, rng: random.Random) -> Image.Image:
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    base = np.stack(
        [
            grid_x,
            grid_y,
            0.5 + 0.5 * np.sin((grid_x + grid_y) * rng.uniform(4, 10)),
        ],
        axis=-1,
    )
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 0.04, size=base.shape)
    image = np.clip(base + noise, 0, 1)
    return Image.fromarray((image * 255).astype(np.uint8), mode="RGB")


def make_anime_like(size: int, rng: random.Random) -> Image.Image:
    bg = tuple(rng.randint(150, 245) for _ in range(3))
    image = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        color = tuple(rng.randint(20, 230) for _ in range(3))
        x0 = rng.randint(0, size - 64)
        y0 = rng.randint(0, size - 64)
        x1 = rng.randint(x0 + 32, min(size, x0 + 220))
        y1 = rng.randint(y0 + 32, min(size, y0 + 220))
        draw.rectangle((x0, y0, x1, y1), fill=color, outline=(10, 10, 10), width=rng.randint(2, 5))
    for _ in range(18):
        color = (10, 10, 10)
        points = [(rng.randint(0, size), rng.randint(0, size)) for _ in range(3)]
        draw.line(points, fill=color, width=rng.randint(1, 4), joint="curve")
    return image


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    images_dir = args.output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        for index in range(args.count):
            domain = "anime" if index % 2 else "photo"
            image = make_anime_like(args.size, rng) if domain == "anime" else make_photo_like(args.size, rng)
            filename = f"{index:04d}.png"
            image.save(images_dir / filename)
            split = "val" if index % 5 == 0 else "train"
            writer.writerow({"path": f"images/{filename}", "domain": domain, "split": split})

    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
