from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


DIV2K_URLS = {
    "train": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "val": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DIV2K HR images and build a photo manifest.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jwheojjang/scratch/sr-diffusion/datasets/photo/div2k"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/jwheojjang/scratch/sr-diffusion/data/manifest_div2k_photo.csv"),
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        if zipfile.is_zipfile(destination):
            print(f"exists: {destination}")
            return
        print(f"invalid zip, re-downloading: {destination}")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()

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
    if not zipfile.is_zipfile(partial):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a valid zip: {url}")
    shutil.move(str(partial), destination)


def extract(zip_path: Path, output_dir: Path) -> None:
    marker = output_dir / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"extracted: {zip_path.name}")
        return
    print(f"extracting {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Not a valid zip file: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    marker.touch()


def build_manifest(output_dir: Path, manifest: Path) -> None:
    rows: list[dict[str, str]] = []
    split_dirs = {
        "train": output_dir / "DIV2K_train_HR",
        "val": output_dir / "DIV2K_valid_HR",
    }
    for split, image_dir in split_dirs.items():
        if not image_dir.exists():
            raise FileNotFoundError(image_dir)
        for path in sorted(image_dir.glob("*.png")):
            rows.append({"path": str(path.resolve()), "domain": "photo", "split": split})

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {manifest} rows={len(rows)}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, url in DIV2K_URLS.items():
        zip_path = args.output_dir / f"DIV2K_{split}_HR.zip"
        if not args.skip_download:
            download(url, zip_path, force=args.force)
        extract(zip_path, args.output_dir)
    build_manifest(args.output_dir, args.manifest)


if __name__ == "__main__":
    main()
