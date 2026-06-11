"""Export RA tensors from 1218-style mmWave MAT files.

This bridges the output of ``build_image_aligned_mmwave_mat.py`` /
``bin_to_1218_mat.py`` to the visualization entrypoint
``visualize_sigle_orig_pred.vis_radar_ra_stream``.

Input:
  - image_to_antframe_time_aligned.csv (must contain a ``mat`` column)
  - directory containing the referenced MAT files

Output:
  - mmwave_ra_npy/*.npy
  - optional bridge CSV with an added ``ra`` column
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


RA_AZIMUTH_RASTER = np.arange(-53.0, 54.0, 1.0, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RA npy files from mmWave MAT files.")
    parser.add_argument("--mapping-csv", type=Path, required=True, help="image_to_antframe_time_aligned.csv path.")
    parser.add_argument("--mat-dir", type=Path, required=True, help="Directory containing referenced MAT files.")
    parser.add_argument("--ra-out-dir", type=Path, required=True, help="Directory to save exported RA npy files.")
    parser.add_argument(
        "--bridge-out-csv",
        type=Path,
        help="Optional CSV path. If set, writes a copy of mapping-csv with an added ra column.",
    )
    return parser.parse_args()


def load_mapping_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "mat" not in fieldnames:
        raise ValueError(f"{path} is missing required column 'mat'")
    return rows, fieldnames


def mat_to_ra_tensor(mat_path: Path) -> np.ndarray:
    data = loadmat(str(mat_path))["Data_Ori"]
    n_el = data.shape[0]
    n_az = len(RA_AZIMUTH_RASTER)

    columns_by_az: list[list[np.ndarray]] = [[] for _ in range(n_az)]
    n_range = 0

    for ei in range(n_el):
        cell = data[ei, 0].ravel()
        az = np.asarray(cell[1]).ravel().astype(np.float64)
        sum_db = np.asarray(cell[3]).astype(np.float32)
        if az.size == 0 or sum_db.size == 0:
            continue
        n_range = max(n_range, sum_db.shape[0])

        # The MAT stores dB-like values. Save amplitude-like values so the
        # existing visualization path (20*log10) yields a sensible heatmap.
        amp = np.power(10.0, sum_db / 20.0)

        az_idx = np.rint(az - RA_AZIMUTH_RASTER[0]).astype(np.int32)
        az_idx = np.clip(az_idx, 0, n_az - 1)
        for src_idx, dst_idx in enumerate(az_idx):
            columns_by_az[int(dst_idx)].append(amp[:, src_idx].astype(np.float32))

    if n_range == 0:
        return np.zeros((0, n_az, 6), dtype=np.float32)

    ra = np.zeros((n_range, n_az, 6), dtype=np.float32)
    for dst_idx, cols in enumerate(columns_by_az):
        if not cols:
            continue
        stacked = np.stack(cols, axis=0)
        ra[:, dst_idx, 0] = np.max(stacked, axis=0)
        ra[:, dst_idx, 1] = np.median(stacked, axis=0)
        ra[:, dst_idx, 2] = np.var(stacked, axis=0)

    return ra


def export_unique_ra(rows: list[dict[str, str]], mat_dir: Path, ra_out_dir: Path) -> dict[str, str]:
    ra_out_dir.mkdir(parents=True, exist_ok=True)
    mat_to_ra_name: dict[str, str] = {}

    unique_mats = sorted({row["mat"] for row in rows if row.get("mat")})
    for mat_name in unique_mats:
        mat_path = mat_dir / mat_name
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        ra_tensor = mat_to_ra_tensor(mat_path)
        ra_name = f"{Path(mat_name).stem}.npy"
        np.save(ra_out_dir / ra_name, ra_tensor, allow_pickle=False)
        mat_to_ra_name[mat_name] = ra_name

    return mat_to_ra_name


def write_bridge_csv(
    out_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    mat_to_ra_name: dict[str, str],
) -> None:
    out_fields = list(fieldnames)
    if "ra" not in out_fields:
        out_fields.append("ra")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            row_out["ra"] = mat_to_ra_name.get(row.get("mat", ""), "")
            writer.writerow(row_out)


def main() -> None:
    args = parse_args()
    rows, fieldnames = load_mapping_rows(args.mapping_csv)
    mat_to_ra_name = export_unique_ra(rows, args.mat_dir, args.ra_out_dir)

    if args.bridge_out_csv:
        write_bridge_csv(args.bridge_out_csv, rows, fieldnames, mat_to_ra_name)

    print(f"mapping csv: {args.mapping_csv}")
    print(f"mat dir: {args.mat_dir}")
    print(f"ra dir: {args.ra_out_dir}")
    print(f"unique mats exported: {len(mat_to_ra_name)}")
    if args.bridge_out_csv:
        print(f"bridge csv with ra column: {args.bridge_out_csv}")


if __name__ == "__main__":
    main()
