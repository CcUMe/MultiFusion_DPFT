#!/usr/bin/env python
"""Draw LabelMe boxes from one camera onto matched visible/IR images.

The dataset layout is expected to contain many ``segment_*/images`` folders,
each with several modality subdirectories. One visible camera directory has
``.json`` files next to its images. For every JSON file, this script draws the
annotated boxes on:

* the annotated source image with the same stem;
* the nearest-timestamp image in each other image subdirectory.

Outputs are written under ``<segment>/label_box_overlays/<modality>/`` by
default. If ``--output-root`` is provided, outputs are written to that separate
directory while preserving the dataset-relative segment structure. Original
images are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TIME_RE = re.compile(r"_t(?P<time>\d+(?:\.\d+)?)")
PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
]


def long_path(path: Path) -> str:
    """Return a Windows long-path-safe string while remaining portable."""
    resolved = str(path.resolve())
    if not resolved.startswith("\\\\?\\") and re.match(r"^[A-Za-z]:\\", resolved):
        return "\\\\?\\" + resolved
    return resolved


def parse_time(path: Path) -> float | None:
    match = TIME_RE.search(path.stem)
    if not match:
        return None
    return float(match.group("time"))


def is_image(path: Path) -> bool:
    return os.path.isfile(long_path(path)) and path.suffix.lower() in IMAGE_EXTS


def path_exists(path: Path) -> bool:
    return os.path.exists(long_path(path))


def color_for_label(label: str) -> tuple[int, int, int]:
    return PALETTE[sum(label.encode("utf-8")) % len(PALETTE)]


def load_shapes(json_path: Path) -> list[dict[str, Any]]:
    with open(long_path(json_path), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("shapes", [])


def shape_bbox(shape: dict[str, Any]) -> tuple[float, float, float, float] | None:
    points = shape.get("points") or []
    if len(points) < 2:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        xs.append(float(point[0]))
        ys.append(float(point[1]))

    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def draw_shapes(
    image_path: Path,
    out_path: Path,
    shapes: list[dict[str, Any]],
    source_size: tuple[int, int],
    line_width: int,
) -> None:
    with Image.open(long_path(image_path)) as im:
        image = im.convert("RGB")

    target_w, target_h = image.size
    source_w, source_h = source_size
    scale_x = target_w / source_w
    scale_y = target_h / source_h

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for shape in shapes:
        label = str(shape.get("label", "unknown"))
        bbox = shape_bbox(shape)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        x1 = clamp(x1 * scale_x, 0, target_w - 1)
        y1 = clamp(y1 * scale_y, 0, target_h - 1)
        x2 = clamp(x2 * scale_x, 0, target_w - 1)
        y2 = clamp(y2 * scale_y, 0, target_h - 1)
        if x2 <= x1 or y2 <= y1:
            continue

        color = color_for_label(label)
        for offset in range(line_width):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)

        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_w = right - left + 6
        label_h = bottom - top + 4
        label_y = y1 - label_h if y1 - label_h >= 0 else y1
        draw.rectangle((x1, label_y, x1 + label_w, label_y + label_h), fill=color)
        draw.text((x1 + 3, label_y + 2), label, fill=(255, 255, 255), font=font)

    os.makedirs(long_path(out_path.parent), exist_ok=True)
    image.save(long_path(out_path), quality=95)


def build_time_index(image_dir: Path) -> list[tuple[float, Path]]:
    indexed: list[tuple[float, Path]] = []
    for image_path in image_dir.iterdir():
        if not is_image(image_path):
            continue
        timestamp = parse_time(image_path)
        if timestamp is not None:
            indexed.append((timestamp, image_path))
    indexed.sort(key=lambda item: item[0])
    return indexed


def nearest_by_time(index: list[tuple[float, Path]], timestamp: float) -> tuple[Path, float] | None:
    if not index:
        return None

    times = [item[0] for item in index]
    pos = bisect_left(times, timestamp)
    candidates = []
    if pos < len(index):
        candidates.append(index[pos])
    if pos > 0:
        candidates.append(index[pos - 1])
    if not candidates:
        return None

    nearest_time, nearest_path = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return nearest_path, abs(nearest_time - timestamp)


def find_label_dirs(images_dir: Path) -> list[Path]:
    return [
        child
        for child in images_dir.iterdir()
        if child.is_dir() and any(child.glob("*.json"))
    ]


def process_images_dir(
    images_dir: Path,
    dataset_root: Path,
    output_root: Path | None,
    output_name: str,
    max_time_diff: float,
    line_width: int,
    overwrite: bool,
    dry_run: bool,
    stats: dict[str, int],
) -> None:
    label_dirs = find_label_dirs(images_dir)
    if not label_dirs:
        stats["images_dirs_without_json"] += 1
        return

    modality_dirs = [child for child in images_dir.iterdir() if child.is_dir()]
    time_indexes = {child: build_time_index(child) for child in modality_dirs}
    if output_root is None:
        segment_output_root = images_dir.parent / output_name
    else:
        segment_rel = images_dir.parent.relative_to(dataset_root)
        segment_output_root = output_root / segment_rel

    for label_dir in label_dirs:
        for json_path in sorted(label_dir.glob("*.json")):
            source_time = parse_time(json_path)
            source_image = next((label_dir / f"{json_path.stem}{ext}" for ext in IMAGE_EXTS if path_exists(label_dir / f"{json_path.stem}{ext}")), None)
            if source_time is None or source_image is None:
                stats["json_without_source_image"] += 1
                continue

            shapes = load_shapes(json_path)
            if not shapes:
                stats["empty_json"] += 1
                continue

            with Image.open(long_path(source_image)) as im:
                source_size = im.size

            for modality_dir in modality_dirs:
                if modality_dir == label_dir:
                    target_image = source_image
                    diff = 0.0
                else:
                    nearest = nearest_by_time(time_indexes[modality_dir], source_time)
                    if nearest is None:
                        stats["missing_modality_match"] += 1
                        continue
                    target_image, diff = nearest
                    if diff > max_time_diff:
                        stats["skipped_by_time_diff"] += 1
                        continue

                out_path = segment_output_root / modality_dir.name / target_image.name
                if out_path.exists() and not overwrite:
                    stats["skipped_existing"] += 1
                    continue

                stats["would_write" if dry_run else "written"] += 1
                if not dry_run:
                    draw_shapes(target_image, out_path, shapes, source_size, line_width)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw LabelMe label boxes on visible and infrared images.")
    parser.add_argument("--root", default="/mnt/disk1/low_altitude/data_exports_annotated_subset/", help="Dataset root directory. Default: current directory.")
    parser.add_argument("--output-name", default="label_box_overlays", help="Output folder name beside each images folder.")
    parser.add_argument(
        "--output-root",
        default="/mnt/disk1/yangqilin/temp/",
        help="Separate output root. Preserves paths relative to --root, for example <output-root>/<capture>/<part>/<segment>/<modality>/<image>.",
    )
    parser.add_argument("--max-time-diff", type=float, default=0.5, help="Maximum timestamp gap in seconds for cross-modality matching.")
    parser.add_argument("--line-width", type=int, default=3, help="Bounding box line width.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing overlay images.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing images.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else None
    stats: dict[str, int] = defaultdict(int)

    images_dirs = sorted(path for path in root.rglob("images") if path.is_dir())
    for images_dir in images_dirs:
        process_images_dir(
            images_dir=images_dir,
            dataset_root=root,
            output_root=output_root,
            output_name=args.output_name,
            max_time_diff=args.max_time_diff,
            line_width=args.line_width,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            stats=stats,
        )

    print(f"images dirs scanned: {len(images_dirs)}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
