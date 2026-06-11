"""End-to-end low-altitude mmWave bin -> image-aligned RAE export.

This version follows the radar protocol more closely:
  - full scans are delimited with start/end flags
  - npy keeps all elevation layers as a 3D RAE tensor
  - visualization metadata still records one representative pitch layer
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import build_image_aligned_mmwave_mat as bridge
import low_altitude_radar_parser as la_parser
import match_radar_camera_anchor as anchor_match
from radar_config import DEFAULT_CONFIG_PATH, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build image-aligned low-altitude mmWave RAE npy files directly from one raw bin capture."
    )
    parser.add_argument("--cap-dir", type=Path, required=True, help="Capture directory containing the bin and *_part folders.")
    parser.add_argument("--bin-path", type=Path, help="Raw *_mmwave_udp.bin path. Defaults to the only such file under cap-dir.")
    parser.add_argument("--packet-csv", type=Path, help="Output packet-level match CSV path.")
    parser.add_argument("--mapping-csv", type=Path, help="Output image-to-scan CSV path.")
    parser.add_argument("--ra-out-dir", type=Path, help="Output directory for matched RAE npy files.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Radar processing config JSON path.")
    parser.add_argument(
        "--export-all-scans",
        action="store_true",
        help="Export every complete scan instead of only those selected by the image mapping CSV.",
    )
    return parser.parse_args()


def write_bridge_csv(out_path: Path, rows: list[dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "camera_seq",
        "camera_t_bag",
        "scan_index",
        "ra",
        "pkt_start",
        "pkt_end",
        "assignment",
        "part_name",
        "scan_t_start",
        "scan_t_end",
        "scan_t_center",
        "selected_pitch_deg",
        "selected_pitch_idx",
        "pitch_layer_count",
        "pitch_layers_deg",
        "scan_dir",
        "az_min_deg",
        "az_max_deg",
        "range_max_m",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_scans(
    scan_ranges: list[la_parser.ScanRange],
    rel_time_out: np.ndarray,
    pkt_part_out: np.ndarray,
) -> list[dict[str, float | int]]:
    scans: list[dict[str, float | int]] = []
    for scan in scan_ranges:
        if not scan.complete:
            continue
        rel_seg = rel_time_out[scan.pkt_start : scan.pkt_end + 1]
        finite = np.isfinite(rel_seg)
        if not finite.any():
            continue
        part_seg = pkt_part_out[scan.pkt_start : scan.pkt_end + 1]
        valid_parts = part_seg[part_seg >= 0]
        if len(valid_parts) == 0:
            continue
        rel_valid = rel_seg[finite]
        scans.append(
            {
                "scan_index": scan.scan_index,
                "pkt_start": scan.pkt_start,
                "pkt_end": scan.pkt_end,
                "part_idx": int(valid_parts[0]),
                "scan_t_start": float(rel_valid[0]),
                "scan_t_end": float(rel_valid[-1]),
                "scan_t_center": float(rel_valid[len(rel_valid) // 2]),
            }
        )
    return scans


def build_image_bridge_rows(parts: list[anchor_match.PartData], scans: list[dict[str, float | int]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    by_part: dict[int, list[dict[str, float | int]]] = {}
    for scan in scans:
        by_part.setdefault(int(scan["part_idx"]), []).append(scan)

    fallback_seq = 0
    for part_idx, part in enumerate(parts):
        part_scans = by_part.get(part_idx, [])
        if not part_scans or len(part.cam_t) == 0:
            continue
        part_scans = sorted(part_scans, key=lambda item: float(item["scan_t_center"]))

        for cam_t, cam_name in zip(part.cam_t, part.cam_n):
            inside = [s for s in part_scans if float(s["scan_t_start"]) <= float(cam_t) <= float(s["scan_t_end"])]
            if inside:
                chosen = min(inside, key=lambda s: abs(float(s["scan_t_center"]) - float(cam_t)))
                assignment = "inside"
            else:
                chosen = min(part_scans, key=lambda s: abs(float(s["scan_t_center"]) - float(cam_t)))
                assignment = "nearest"

            rows.append(
                {
                    "image": cam_name,
                    "camera_seq": str(bridge.parse_camera_seq(cam_name, fallback_seq)),
                    "camera_t_bag": f"{float(cam_t):.6f}",
                    "scan_index": str(int(chosen["scan_index"])),
                    "ra": "",
                    "pkt_start": str(int(chosen["pkt_start"])),
                    "pkt_end": str(int(chosen["pkt_end"])),
                    "assignment": assignment,
                    "part_name": part.name,
                    "scan_t_start": f"{float(chosen['scan_t_start']):.6f}",
                    "scan_t_end": f"{float(chosen['scan_t_end']):.6f}",
                    "scan_t_center": f"{float(chosen['scan_t_center']):.6f}",
                    "selected_pitch_deg": "",
                    "selected_pitch_idx": "",
                    "pitch_layer_count": "",
                    "pitch_layers_deg": "",
                    "scan_dir": "",
                    "az_min_deg": "",
                    "az_max_deg": "",
                    "range_max_m": "",
                }
            )
            fallback_seq += 1

    rows.sort(key=lambda row: (float(row["camera_t_bag"]), int(row["camera_seq"])))
    return rows


def export_scan_ra(
    bin_path: Path,
    ra_out_dir: Path,
    scan_ranges: list[la_parser.ScanRange],
    pitch_cfg: dict,
    selected_scan_indices: set[int] | None = None,
) -> tuple[dict[int, str], dict[int, dict[str, float | int | str]]]:
    ra_out_dir.mkdir(parents=True, exist_ok=True)
    ra_names: dict[int, str] = {}
    scan_meta: dict[int, dict[str, float | int | str]] = {}

    for scan in scan_ranges:
        if not scan.complete:
            continue
        if selected_scan_indices is not None and scan.scan_index not in selected_scan_indices:
            continue

        arr = la_parser.read_packet_block(bin_path, scan.pkt_start, scan.pkt_end)
        block = la_parser.decode_scan_block(arr)
        chosen_el, _ = la_parser.choose_pitch_layer(block["ant_el"], pitch_cfg)

        sum_db_666 = la_parser.power_to_db(block["sum_lin"])[:, 1 : 1 + la_parser.N_RANGE_OUT]
        rae, az_values, pitch_values = la_parser.build_ra_volume(
            sum_db_666,
            block["ant_az"],
            block["ant_el"],
            pitch_cfg,
        )
        if rae.size == 0 or pitch_values.size == 0:
            continue

        selected_pitch_idx = int(np.argmin(np.abs(pitch_values - chosen_el)))
        meta = la_parser.scan_meta_from_selected(
            block["range_km"],
            az_values,
            chosen_el,
            selected_pitch_idx,
            pitch_values,
            block["scan_dir"],
        )

        ra_name = f"{bin_path.stem}_Scan{scan.scan_index:03d}_FZ{scan.pkt_start:06d}-{scan.pkt_end:06d}.npy"
        np.save(ra_out_dir / ra_name, rae, allow_pickle=False)
        ra_names[scan.scan_index] = ra_name
        scan_meta[scan.scan_index] = meta

    return ra_names, scan_meta


def populate_bridge_rows(rows: list[dict[str, str]], ra_names: dict[int, str], scan_meta: dict[int, dict[str, float | int | str]]) -> None:
    for row in rows:
        scan_index = int(row["scan_index"])
        row["ra"] = ra_names.get(scan_index, "")
        meta = scan_meta.get(scan_index)
        if meta is not None:
            row["selected_pitch_deg"] = f"{float(meta['selected_pitch_deg']):.2f}"
            row["selected_pitch_idx"] = str(int(meta["selected_pitch_idx"]))
            row["pitch_layer_count"] = str(int(meta["pitch_layer_count"]))
            row["pitch_layers_deg"] = str(meta["pitch_layers_deg"])
            row["scan_dir"] = str(int(meta["scan_dir"]))
            row["az_min_deg"] = f"{float(meta['az_min_deg']):.3f}"
            row["az_max_deg"] = f"{float(meta['az_max_deg']):.3f}"
            row["range_max_m"] = f"{float(meta['range_max_m']):.3f}"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_cfg = cfg["ra_output"]

    cap_dir = args.cap_dir.resolve()
    bin_path = bridge.resolve_bin_path(cap_dir, args.bin_path.resolve() if args.bin_path else None)

    packet_csv = args.packet_csv.resolve() if args.packet_csv else bin_path.parent / output_cfg["packet_csv_name"]
    mapping_csv = args.mapping_csv.resolve() if args.mapping_csv else bin_path.parent / output_cfg["mapping_csv_name"]
    ra_out_dir = args.ra_out_dir.resolve() if args.ra_out_dir else bin_path.parent / output_cfg["ra_dir_name"]

    parts = anchor_match.load_capture(cap_dir)
    if not parts:
        raise RuntimeError(f"no *_part capture folders with nav100/image data found under {cap_dir}")

    antframe_arr, rel_time_out, pkt_part_out, cam_names_out, cam_rt_out = bridge.build_packet_alignment(bin_path, parts)
    bridge.write_packet_csv(packet_csv, antframe_arr, rel_time_out, cam_names_out, cam_rt_out)

    scan_ranges = la_parser.build_scan_ranges(bin_path)
    scans = summarize_scans(scan_ranges, rel_time_out, pkt_part_out)
    bridge_rows = build_image_bridge_rows(parts, scans)
    if not bridge_rows:
        raise RuntimeError("no image rows could be matched to any complete scan")

    if args.export_all_scans:
        selected_scan_indices = None
    else:
        selected_scan_indices = {int(row["scan_index"]) for row in bridge_rows}

    ra_names, scan_meta = export_scan_ra(bin_path, ra_out_dir, scan_ranges, cfg["pitch_selection"], selected_scan_indices)
    populate_bridge_rows(bridge_rows, ra_names, scan_meta)
    write_bridge_csv(mapping_csv, bridge_rows)

    print(f"config: {args.config.resolve()}")
    print(f"bin: {bin_path}")
    print(f"parts: {len(parts)}")
    print(f"packet csv: {packet_csv}")
    print(f"bridge csv: {mapping_csv}")
    print(f"ra dir: {ra_out_dir}")
    print(f"complete scans: {sum(1 for s in scan_ranges if s.complete)}")
    print(f"matched images: {len(bridge_rows)}")
    print(f"exported ra npy: {len(ra_names)}")


if __name__ == "__main__":
    main()
