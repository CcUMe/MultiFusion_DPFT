#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraBlock:
    title: str
    camera_name: str
    width: int | None
    height: int | None
    camera_matrix: list[list[float]]
    distortion: list[float]
    rectification: list[list[float]]
    projection: list[list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a combined calibration text into per-camera txt files usable by current radar-camera scripts."
    )
    parser.add_argument("input_txt", type=Path, help="Combined calibration text, for example 5-19.txt")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory. Defaults to the input file parent directory.",
    )
    return parser.parse_args()


def _collect_nonempty(lines: list[str], start: int, count: int) -> tuple[list[str], int]:
    vals: list[str] = []
    idx = start
    while idx < len(lines) and len(vals) < count:
        text = lines[idx].strip()
        if text:
            vals.append(text)
        idx += 1
    if len(vals) < count:
        raise ValueError(f"expected {count} non-empty lines starting from line {start + 1}")
    return vals, idx


def _parse_matrix(lines: list[str], idx: int, rows: int) -> tuple[list[list[float]], int]:
    raw_rows, next_idx = _collect_nonempty(lines, idx, rows)
    matrix = [[float(x) for x in row.split()] for row in raw_rows]
    return matrix, next_idx


def _parse_vector(lines: list[str], idx: int) -> tuple[list[float], int]:
    raw_rows, next_idx = _collect_nonempty(lines, idx, 1)
    return [float(x) for x in raw_rows[0].split()], next_idx


def parse_combined_calib(path: Path) -> list[CameraBlock]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: list[CameraBlock] = []
    idx = 0
    current_title = ""
    current_width: int | None = None
    current_height: int | None = None

    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if re.match(r"^\d[\d\- ]*(左|右)$", line):
            current_title = line
            idx += 1
            continue
        if line.lower() == "width":
            raw, idx = _collect_nonempty(lines, idx + 1, 1)
            current_width = int(float(raw[0]))
            continue
        if line.lower() == "height":
            raw, idx = _collect_nonempty(lines, idx + 1, 1)
            current_height = int(float(raw[0]))
            continue
        if line.startswith("[") and line.endswith("]"):
            camera_name = line[1:-1].strip()
            if camera_name.lower() == "image":
                idx += 1
                continue
            camera_matrix: list[list[float]] = []
            distortion: list[float] = []
            rectification: list[list[float]] = []
            projection: list[list[float]] = []
            idx += 1
            while idx < len(lines):
                key = lines[idx].strip().lower()
                if not key:
                    idx += 1
                    continue
                if key.startswith("[") and key.endswith("]"):
                    break
                if re.match(r"^\d[\d\- ]*(左|右)$", key):
                    break
                if key == "camera matrix":
                    camera_matrix, idx = _parse_matrix(lines, idx + 1, 3)
                    continue
                if key == "distortion":
                    distortion, idx = _parse_vector(lines, idx + 1)
                    continue
                if key == "rectification":
                    rectification, idx = _parse_matrix(lines, idx + 1, 3)
                    continue
                if key == "projection":
                    projection, idx = _parse_matrix(lines, idx + 1, 3)
                    continue
                if key == "width":
                    raw, idx = _collect_nonempty(lines, idx + 1, 1)
                    current_width = int(float(raw[0]))
                    continue
                if key == "height":
                    raw, idx = _collect_nonempty(lines, idx + 1, 1)
                    current_height = int(float(raw[0]))
                    continue
                idx += 1
            blocks.append(
                CameraBlock(
                    title=current_title,
                    camera_name=camera_name,
                    width=current_width,
                    height=current_height,
                    camera_matrix=camera_matrix,
                    distortion=distortion,
                    rectification=rectification,
                    projection=projection,
                )
            )
            continue
        idx += 1
    return blocks


def infer_side(title: str, camera_name: str) -> str | None:
    text = f"{title} {camera_name}"
    if "右" in text:
        return "right"
    if "左" in text:
        return "left"
    return None


def ensure_defaults(block: CameraBlock) -> CameraBlock:
    if not block.rectification:
        block.rectification = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if not block.projection and block.camera_matrix:
        fx = block.camera_matrix[0][0]
        fy = block.camera_matrix[1][1]
        cx = block.camera_matrix[0][2]
        cy = block.camera_matrix[1][2]
        block.projection = [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    return block


def format_block(block: CameraBlock) -> str:
    lines: list[str] = []
    if block.title:
        lines.append(block.title)
        lines.append("")
    lines.append("# oST version 5.0 parameters")
    lines.append("")
    lines.append("[image]")
    lines.append("")
    if block.width is not None:
        lines.append("width")
        lines.append(str(block.width))
        lines.append("")
    if block.height is not None:
        lines.append("height")
        lines.append(str(block.height))
        lines.append("")
    lines.append(f"[{block.camera_name}]")
    lines.append("")
    lines.append("camera matrix")
    for row in block.camera_matrix:
        lines.append(" ".join(f"{v:.6f}" for v in row))
    lines.append("")
    lines.append("distortion")
    lines.append(" ".join(f"{v:.6f}" for v in block.distortion))
    lines.append("")
    lines.append("rectification")
    for row in block.rectification:
        lines.append(" ".join(f"{v:.6f}" for v in row))
    lines.append("")
    lines.append("projection")
    for row in block.projection:
        lines.append(" ".join(f"{v:.6f}" for v in row))
    lines.append("")
    return "\n".join(lines)


def write_outputs(blocks: list[CameraBlock], out_dir: Path) -> list[Path]:
    written: list[Path] = []
    for block in blocks:
        block = ensure_defaults(block)
        side = infer_side(block.title, block.camera_name)
        text = format_block(block)
        camera_file = out_dir / f"{block.camera_name}.txt"
        camera_file.write_text(text, encoding="utf-8")
        written.append(camera_file)
        if side is not None:
            side_file = out_dir / f"{side}cam.txt"
            side_file.write_text(text, encoding="utf-8")
            written.append(side_file)
    return written


def main() -> None:
    args = parse_args()
    input_txt = args.input_txt.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else input_txt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = parse_combined_calib(input_txt)
    if not blocks:
        raise SystemExit(f"no camera blocks parsed from {input_txt}")
    missing = [b.camera_name for b in blocks if not b.projection and not b.camera_matrix]
    if missing:
        raise SystemExit(f"camera blocks missing projection section: {', '.join(missing)}")

    written = write_outputs(blocks, out_dir)
    print(f"input: {input_txt}")
    print(f"output dir: {out_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
