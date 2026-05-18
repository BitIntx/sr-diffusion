from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path


FLICKR2K_URL = "https://huggingface.co/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Flickr2K and extract HR images only.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jwheojjang/scratch/sr-diffusion/datasets/photo/flickr2k"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/jwheojjang/scratch/sr-diffusion/data/manifest_flickr2k_photo.csv"),
    )
    parser.add_argument("--url", type=str, default=FLICKR2K_URL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    return parser.parse_args()


def download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        if is_valid_archive(destination):
            print(f"exists: {destination}")
            return
        print(f"invalid tar, re-downloading: {destination}")
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
        if not is_valid_archive(partial):
            raise RuntimeError(f"Downloaded file is not a valid archive: {url}")
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
    if not is_valid_archive(partial):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a valid archive: {url}")
    shutil.move(str(partial), destination)


def is_valid_archive(path: Path) -> bool:
    return tarfile.is_tarfile(path) or zipfile.is_zipfile(path)


def is_hr_member(name: str) -> bool:
    path = Path(name)
    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        return False
    parts = [part.lower() for part in path.parts]
    if any("lr" in part for part in parts):
        return False
    if any("hr" in part for part in parts):
        return True
    # Some mirrors contain only HR files directly under Flickr2K.
    return any(part == "flickr2k" for part in parts)


def extract_hr_images(archive_path: Path, output_dir: Path) -> Path:
    if not is_valid_archive(archive_path):
        raise RuntimeError(f"Not a valid archive file: {archive_path}")

    hr_dir = output_dir / "Flickr2K_HR"
    marker = output_dir / ".extracted_Flickr2K_HR"
    if marker.exists() and len(list(hr_dir.glob("*.*"))) >= 2650:
        print(f"extracted: {hr_dir}")
        return hr_dir

    print(f"extracting HR images from {archive_path}")
    hr_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if is_hr_member(name)]
            if not members:
                raise RuntimeError("No HR image members found in Flickr2K archive")
            for name in members:
                out_path = hr_dir / Path(name).name
                with archive.open(name) as source, out_path.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
                count += 1
                if count % 250 == 0:
                    print(f"extracted {count} images")
    else:
        with tarfile.open(archive_path) as archive:
            members = [member for member in archive.getmembers() if member.isfile() and is_hr_member(member.name)]
            if not members:
                raise RuntimeError("No HR image members found in Flickr2K archive")
            for member in members:
                source = archive.extractfile(member)
                if source is None:
                    continue
                out_path = hr_dir / Path(member.name).name
                with out_path.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
                count += 1
                if count % 250 == 0:
                    print(f"extracted {count} images")
    marker.write_text(str(count), encoding="utf-8")
    print(f"extracted {count} images to {hr_dir}")
    return hr_dir


def build_manifest(hr_dir: Path, manifest: Path) -> None:
    paths = [path for path in sorted(hr_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
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
    archive_suffix = ".zip" if args.url.lower().split("?")[0].endswith(".zip") else ".tar"
    archive = args.output_dir / f"Flickr2K{archive_suffix}"
    if not args.skip_download:
        download(args.url, archive, force=args.force)
    hr_dir = extract_hr_images(archive, args.output_dir)
    build_manifest(hr_dir, args.manifest)
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"removed archive {archive}")


if __name__ == "__main__":
    main()
