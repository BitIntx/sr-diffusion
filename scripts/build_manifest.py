from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a photo/anime manifest CSV.")
    parser.add_argument("--photo-dir", type=Path, required=True)
    parser.add_argument("--anime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Write absolute image paths instead of paths relative to the manifest directory.",
    )
    return parser.parse_args()


def scan_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    images = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images)


def display_path(path: Path, manifest_dir: Path, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(manifest_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")

    rows: list[dict[str, str]] = []
    rng = random.Random(args.seed)
    manifest_dir = args.output.parent

    for domain, root in [("photo", args.photo_dir), ("anime", args.anime_dir)]:
        paths = scan_images(root)
        rng.shuffle(paths)
        val_count = int(round(len(paths) * args.val_fraction))
        val_paths = set(paths[:val_count])
        for path in paths:
            rows.append(
                {
                    "path": display_path(path, manifest_dir, args.absolute),
                    "domain": domain,
                    "split": "val" if path in val_paths else "train",
                }
            )
        print(f"{domain}: {len(paths)} images, val={val_count}")

    rows.sort(key=lambda row: (row["split"], row["domain"], row["path"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
