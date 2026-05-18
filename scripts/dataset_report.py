from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print basic manifest image statistics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Limit image inspection count; 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)

    domain_split = Counter((row["domain"], row["split"]) for row in rows)
    print(f"manifest={args.manifest}")
    print(f"rows={len(rows)}")
    for (domain, split), count in sorted(domain_split.items()):
        print(f"{domain}/{split}: {count}")

    inspected = rows if args.limit <= 0 else rows[: args.limit]
    size_by_split: dict[str, list[tuple[int, int]]] = defaultdict(list)
    failures = 0
    for row in inspected:
        try:
            with Image.open(row["path"]) as image:
                size_by_split[row["split"]].append(image.size)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"failed: {row['path']} {exc}")

    for split, sizes in sorted(size_by_split.items()):
        widths = [size[0] for size in sizes]
        heights = [size[1] for size in sizes]
        print(
            f"{split}: inspected={len(sizes)} "
            f"min={min(widths)}x{min(heights)} "
            f"max={max(widths)}x{max(heights)}"
        )
    if failures:
        print(f"failures={failures}")


if __name__ == "__main__":
    main()
