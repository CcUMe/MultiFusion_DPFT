"""Batch process low-altitude captures under one root directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from radar_config import DEFAULT_CONFIG_PATH, load_config


SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = SCRIPT_DIR / "build_image_aligned_ra.py"
VIS_SCRIPT = SCRIPT_DIR / "visualize_low_altitude_ra_pairs.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively process all low-altitude capture subdirectories under one root."
    )
    parser.add_argument("--root-dir", type=Path, required=True, help="Root directory to scan recursively for *_mmwave_udp.bin files.")
    parser.add_argument("--preview-dir-name", help="Per-capture preview directory name. Defaults to config value.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Radar processing config JSON path.")
    parser.add_argument("--max-captures", type=int, default=0, help="Optional limit on processed captures. 0 means all.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip captures whose preview compare directory already contains PNG files.")
    return parser.parse_args()


def find_capture_dirs(root_dir: Path) -> list[Path]:
    return sorted({p.parent.resolve() for p in root_dir.rglob("*_mmwave_udp.bin")})


def should_skip(cap_dir: Path, preview_dir_name: str, compare_subdir_name: str) -> bool:
    compare_dir = cap_dir / preview_dir_name / compare_subdir_name
    return compare_dir.exists() and any(compare_dir.glob("*.png"))


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def process_capture(cap_dir: Path, config_path: Path, output_cfg: dict, vis_cfg: dict, preview_dir_name: str) -> None:
    mapping_csv = cap_dir / output_cfg["mapping_csv_name"]
    ra_dir = cap_dir / output_cfg["ra_dir_name"]
    preview_dir = cap_dir / preview_dir_name

    run_cmd(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--cap-dir",
            str(cap_dir),
            "--config",
            str(config_path),
        ],
        cwd=SCRIPT_DIR,
    )

    run_cmd(
        [
            sys.executable,
            str(VIS_SCRIPT),
            "--cap-dir",
            str(cap_dir),
            "--mapping-csv",
            str(mapping_csv),
            "--ra-dir",
            str(ra_dir),
            "--preview-dir",
            str(preview_dir),
            "--config",
            str(config_path),
        ],
        cwd=SCRIPT_DIR,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_cfg = cfg["ra_output"]
    vis_cfg = cfg["visualization"]
    preview_dir_name = args.preview_dir_name or vis_cfg["preview_dir_name"]

    root_dir = args.root_dir.resolve()
    capture_dirs = find_capture_dirs(root_dir)

    if args.max_captures > 0:
        capture_dirs = capture_dirs[:args.max_captures]

    print(f"config: {args.config.resolve()}")
    print(f"root dir: {root_dir}")
    print(f"found capture dirs: {len(capture_dirs)}")

    processed = 0
    skipped = 0
    failed = 0

    for idx, cap_dir in enumerate(capture_dirs, start=1):
        print(f"\n[{idx}/{len(capture_dirs)}] {cap_dir}")
        if args.skip_existing and should_skip(cap_dir, preview_dir_name, vis_cfg["compare_subdir_name"]):
            print("  skip: existing compare PNGs found")
            skipped += 1
            continue
        try:
            process_capture(cap_dir, args.config.resolve(), output_cfg, vis_cfg, preview_dir_name)
            processed += 1
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"  failed: command exited with {exc.returncode}")
        except Exception as exc:
            failed += 1
            print(f"  failed: {exc}")

    print("\nsummary")
    print(f"processed: {processed}")
    print(f"skipped: {skipped}")
    print(f"failed: {failed}")


if __name__ == "__main__":
    main()
