from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image


COCO_TRAIN2017_URL = "http://images.cocodataset.org/zips/train2017.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download COCO train2017 and extract a deterministic photo subset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jwheojjang/scratch/sr-diffusion/datasets/photo/coco2017"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/jwheojjang/scratch/sr-diffusion/data/manifest_coco2017_photo.csv"),
    )
    parser.add_argument("--url", type=str, default=COCO_TRAIN2017_URL)
    parser.add_argument("--target-count", type=int, default=6550)
    parser.add_argument("--min-size", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    return parser.parse_args()


def is_valid_zip(path: Path) -> bool:
    return zipfile.is_zipfile(path)


def download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        if is_valid_zip(destination):
            print(f"exists: {destination}")
            return
        print(f"invalid zip, re-downloading: {destination}")
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if force:
        partial.unlink(missing_ok=True)

    if shutil.which("curl"):
        print(f"downloading {url}")
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "5",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
        if not is_valid_zip(partial):
            raise RuntimeError(f"Downloaded file is not a valid zip: {url}")
        shutil.move(str(partial), destination)
        return

    partial.unlink(missing_ok=True)
    last_report = [0.0]

    def report(blocks: int, block_size: int, total_size: int) -> None:
        now = time.monotonic()
        downloaded = blocks * block_size
        if total_size > 0 and downloaded < total_size and now - last_report[0] < 1.0:
            return
        last_report[0] = now
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            sys.stdout.write(f"\r{destination.name}: {downloaded / 1e9:.2f}/{total_size / 1e9:.2f} GB {pct:5.1f}%")
        else:
            sys.stdout.write(f"\r{destination.name}: {downloaded / 1e9:.2f} GB")
        sys.stdout.flush()

    print(f"downloading {url}")
    urllib.request.urlretrieve(url, partial, reporthook=report)
    print()
    if not is_valid_zip(partial):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a valid zip: {url}")
    shutil.move(str(partial), destination)


def image_members(archive: zipfile.ZipFile) -> list[str]:
    members = []
    for name in archive.namelist():
        path = Path(name)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not any(part == "train2017" for part in path.parts):
            continue
        members.append(name)
    return members


def image_is_large_enough(archive: zipfile.ZipFile, name: str, min_size: int) -> bool:
    try:
        with archive.open(name) as handle:
            with Image.open(handle) as image:
                width, height = image.size
        return min(width, height) >= min_size
    except Exception as exc:
        print(f"skip unreadable image {name}: {exc}")
        return False


def extract_subset(archive_path: Path, output_dir: Path, target_count: int, min_size: int, seed: int) -> Path:
    if not is_valid_zip(archive_path):
        raise RuntimeError(f"Not a valid zip file: {archive_path}")

    subset_dir = output_dir / "train2017_subset"
    marker = output_dir / f".extracted_train2017_subset_{target_count}_min{min_size}_seed{seed}"
    subset_dir.mkdir(parents=True, exist_ok=True)

    existing = [path for path in subset_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if marker.exists() and len(existing) >= target_count:
        print(f"extracted: {subset_dir} rows={len(existing)}")
        return subset_dir

    print(f"extracting {target_count} COCO train2017 images from {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        members = image_members(archive)
        rng = random.Random(seed)
        rng.shuffle(members)

        count = len(existing)
        existing_names = {path.name for path in existing}
        for name in members:
            if count >= target_count:
                break
            out_path = subset_dir / Path(name).name
            if out_path.name in existing_names:
                continue
            if min_size > 0 and not image_is_large_enough(archive, name, min_size):
                continue
            with archive.open(name) as source, out_path.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            count += 1
            if count % 250 == 0 or count == target_count:
                print(f"extracted {count}/{target_count} images")

    if count < target_count:
        raise RuntimeError(f"Only extracted {count} images; target was {target_count}")
    marker.write_text(str(count), encoding="utf-8")
    print(f"extracted {count} images to {subset_dir}")
    return subset_dir


def build_manifest(image_dir: Path, manifest: Path) -> None:
    paths = [path for path in sorted(image_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        for path in paths:
            writer.writerow({"path": str(path.resolve()), "domain": "photo", "split": "train"})
    print(f"wrote {manifest} rows={len(paths)}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "train2017.zip"
    if not args.skip_download:
        download(args.url, archive, force=args.force)
    image_dir = extract_subset(
        archive,
        args.output_dir,
        target_count=args.target_count,
        min_size=args.min_size,
        seed=args.seed,
    )
    build_manifest(image_dir, args.manifest)
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"removed archive {archive}")


if __name__ == "__main__":
    main()
