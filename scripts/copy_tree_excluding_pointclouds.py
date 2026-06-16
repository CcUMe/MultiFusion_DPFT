#!/usr/bin/env python3
"""Copy a directory tree while excluding directories named `pointclouds`."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


EXCLUDED_DIRNAME = "pointclouds"


@dataclass
class CopyStats:
    total_bytes: int = 0
    total_files: int = 0
    excluded_bytes: int = 0
    excluded_files: int = 0
    copied_bytes: int = 0
    copied_files: int = 0


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def directory_size(path: Path) -> tuple[int, int]:
    total_bytes = 0
    total_files = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                if file_path.is_symlink():
                    continue
                total_bytes += file_path.stat().st_size
                total_files += 1
            except OSError as exc:
                print(f"[warn] Failed to stat {file_path}: {exc}", file=sys.stderr)
    return total_bytes, total_files


def scan_source(source: Path, excluded_dirname: str) -> CopyStats:
    stats = CopyStats()

    for root, dirs, files in os.walk(source):
        excluded_dirs = [name for name in dirs if name == excluded_dirname]
        for excluded_name in excluded_dirs:
            excluded_path = Path(root) / excluded_name
            excluded_bytes, excluded_files = directory_size(excluded_path)
            stats.excluded_bytes += excluded_bytes
            stats.excluded_files += excluded_files

        dirs[:] = [name for name in dirs if name != excluded_dirname]

        for name in files:
            file_path = Path(root) / name
            try:
                if file_path.is_symlink():
                    continue
                stats.total_bytes += file_path.stat().st_size
                stats.total_files += 1
            except OSError as exc:
                print(f"[warn] Failed to stat {file_path}: {exc}", file=sys.stderr)

    return stats


def copy_tree(source: Path, destination: Path, excluded_dirname: str, stats: CopyStats) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(source):
        dirs[:] = [name for name in dirs if name != excluded_dirname]

        root_path = Path(root)
        relative_root = root_path.relative_to(source)
        target_root = destination / relative_root
        target_root.mkdir(parents=True, exist_ok=True)

        for name in files:
            src_file = root_path / name
            dst_file = target_root / name

            try:
                if src_file.is_symlink():
                    continue
                shutil.copy2(src_file, dst_file)
                file_size = src_file.stat().st_size
            except OSError as exc:
                print(f"[warn] Failed to copy {src_file} -> {dst_file}: {exc}", file=sys.stderr)
                continue

            stats.copied_files += 1
            stats.copied_bytes += file_size

            print(
                f"[copy] {stats.copied_files}/{stats.total_files} files, "
                f"{format_bytes(stats.copied_bytes)}/{format_bytes(stats.total_bytes)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a directory tree while excluding directories named `pointclouds`."
    )
    parser.add_argument("source", type=Path, help="Source directory")
    parser.add_argument("destination", type=Path, help="Destination directory")
    parser.add_argument(
        "--exclude-dirname",
        default=EXCLUDED_DIRNAME,
        help=f"Directory name to exclude (default: {EXCLUDED_DIRNAME})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only calculate sizes, do not copy files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()

    if not source.exists():
        print(f"[error] Source does not exist: {source}", file=sys.stderr)
        return 1
    if not source.is_dir():
        print(f"[error] Source is not a directory: {source}", file=sys.stderr)
        return 1

    stats = scan_source(source, args.exclude_dirname)

    print(f"Source      : {source}")
    print(f"Destination : {destination}")
    print(f"Excluded dir: {args.exclude_dirname}")
    print(f"Will copy   : {stats.total_files} files, {format_bytes(stats.total_bytes)}")
    print(f"Will skip   : {stats.excluded_files} files, {format_bytes(stats.excluded_bytes)}")

    if args.dry_run:
        print("Dry run only, no files copied.")
        return 0

    copy_tree(source, destination, args.exclude_dirname, stats)
    print("Copy finished.")
    print(f"Copied      : {stats.copied_files} files, {format_bytes(stats.copied_bytes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
