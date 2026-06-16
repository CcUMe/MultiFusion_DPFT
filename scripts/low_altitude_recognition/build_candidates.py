from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from common import choose_layer_index, compute_range_azimuth, load_csv_rows, power_to_db, safe_float, safe_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pure-radar candidates for low-altitude traditional recognition.")
    parser.add_argument("--cap-dir", type=Path, required=True, help="Capture directory with image_to_antframe_time_aligned.csv and mmwave_ra_npy")
    parser.add_argument("--out-dir", type=Path, help="Output candidate directory. Defaults to <cap-dir>/recognition_candidates")
    parser.add_argument("--mapping-csv", type=Path, help="Aligned mapping CSV. Defaults to <cap-dir>/image_to_antframe_time_aligned.csv")
    parser.add_argument("--ra-dir", type=Path, help="RAE npy directory. Defaults to <cap-dir>/mmwave_ra_npy")
    parser.add_argument("--candidate-source", choices=["fusion", "protocol", "rae"], default="fusion", help="Candidate source: protocol, rae, or fusion of both")
    parser.add_argument("--protocol-root", type=Path, help="Protocol parse output root. Defaults to <cap-dir>/protocol_parse_out or <cap-dir>/protocol_parse_out/scans")
    parser.add_argument("--max-scans", type=int, default=0, help="Optional scan limit. 0 means all.")
    parser.add_argument("--min-area-px", type=int, default=40, help="Minimum connected-component area in radar pixels.")
    parser.add_argument("--min-peak-db", type=float, default=None, help="Optional absolute dB threshold. If omitted, uses percentile threshold.")
    parser.add_argument("--threshold-percentile", type=float, default=99.2, help="Percentile threshold on selected RA slice.")
    parser.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian smoothing sigma before connected components.")
    parser.add_argument("--max-candidates-per-scan", type=int, default=20, help="Maximum kept candidates per scan after pruning.")
    parser.add_argument("--save-preview-json", action="store_true", help="Save one per-scan candidate preview json for debugging")
    return parser.parse_args()


def choose_best_row_per_scan(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    total_rows = len(rows)
    for row_idx, row in enumerate(rows, start=1):
        if row.get("scan_index") and row.get("ra"):
            grouped[row["scan_index"]].append(row)

    selected: list[dict[str, str]] = []
    for items in grouped.values():
        def score(row: dict[str, str]) -> tuple[float, float, str]:
            center = float(row.get("scan_t_center", row.get("camera_t_bag", "0")) or 0.0)
            cam_t = float(row.get("camera_t_bag", "0") or 0.0)
            return (abs(cam_t - center), cam_t, row.get("image", ""))
        selected.append(min(items, key=score))
    selected.sort(key=lambda row: int(row["scan_index"]))
    return selected


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cap_dir = args.cap_dir.resolve()
    mapping_csv = args.mapping_csv.resolve() if args.mapping_csv else cap_dir / "image_to_antframe_time_aligned.csv"
    ra_dir = args.ra_dir.resolve() if args.ra_dir else cap_dir / "mmwave_ra_npy"
    out_dir = args.out_dir.resolve() if args.out_dir else cap_dir / "recognition_candidates"
    return mapping_csv, ra_dir, out_dir


def load_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return load_csv_rows(path)


def resolve_protocol_scans_root(cap_dir: Path, protocol_root: Path | None) -> Path:
    if protocol_root is not None:
        root = protocol_root.resolve()
    else:
        root = (cap_dir / "protocol_parse_out").resolve()
    if (root / "scans").exists():
        return root / "scans"
    return root


def load_protocol_scan_records(scans_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cursor = 0
    scan_dirs = sorted([p for p in scans_root.iterdir() if p.is_dir() and p.name.startswith("scan_")]) if scans_root.exists() else []
    for scan_dir in scan_dirs:
        summary_path = scan_dir / "scan_summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
        frame_count = safe_int(summary.get("frame_count"), 0)
        pkt_start = cursor
        pkt_end = cursor + frame_count - 1 if frame_count > 0 else cursor - 1
        if frame_count > 0:
            cursor = pkt_end + 1
        records.append({
            "path": scan_dir,
            "dir_scan_index": safe_int(scan_dir.name.split("_")[-1], -1),
            "summary_scan_index": safe_int(summary.get("scan_index"), -1),
            "frame_count": frame_count,
            "pkt_start": pkt_start,
            "pkt_end": pkt_end,
        })
    return records


def find_protocol_scan_dir(scan_records: list[dict[str, Any]], scan_row: dict[str, str]) -> Path | None:
    pkt_start = safe_int(scan_row.get("pkt_start"), -1)
    pkt_end = safe_int(scan_row.get("pkt_end"), -1)
    scan_index = safe_int(scan_row.get("scan_index"), -1)

    if pkt_start >= 0 and pkt_end >= pkt_start:
        for rec in scan_records:
            if rec["pkt_start"] == pkt_start and rec["pkt_end"] == pkt_end:
                return rec["path"]

    if scan_index >= 0:
        for rec in scan_records:
            if rec["dir_scan_index"] == scan_index or rec["summary_scan_index"] == scan_index:
                return rec["path"]

    if pkt_start >= 0:
        for rec in scan_records:
            if rec["pkt_start"] <= pkt_start <= rec["pkt_end"]:
                return rec["path"]

    return None


def xyz_nwu_to_read(x_north_m: float, y_west_m: float, z_up_m: float) -> dict[str, float]:
    r_xy = math.hypot(x_north_m, y_west_m)
    r = math.sqrt(x_north_m ** 2 + y_west_m ** 2 + z_up_m ** 2)
    elevation_deg = 0.0 if r == 0 else math.degrees(math.atan2(z_up_m, r_xy))
    azimuth_deg = math.degrees(math.atan2(y_west_m, x_north_m))
    return {
        "range_m": r,
        "elevation_deg": elevation_deg,
        "azimuth_deg": azimuth_deg,
    }


def xyz_nwu_to_body_read(x_north_m: float, y_west_m: float, z_up_m: float, heading_deg: float | None) -> dict[str, float]:
    if heading_deg is None or not math.isfinite(float(heading_deg)):
        return xyz_nwu_to_read(x_north_m, y_west_m, z_up_m)
    h = math.radians(float(heading_deg))
    x_forward_m = x_north_m * math.cos(h) - y_west_m * math.sin(h)
    y_left_m = x_north_m * math.sin(h) + y_west_m * math.cos(h)
    r_xy = math.hypot(x_forward_m, y_left_m)
    r = math.sqrt(x_forward_m ** 2 + y_left_m ** 2 + z_up_m ** 2)
    elevation_deg = 0.0 if r == 0 else math.degrees(math.atan2(z_up_m, r_xy))
    azimuth_deg = math.degrees(math.atan2(y_left_m, x_forward_m))
    return {
        "range_m": r,
        "elevation_deg": elevation_deg,
        "azimuth_deg": azimuth_deg,
    }


def read_to_range_idx(range_m: float, n_range: int, range_max_m: float) -> float:
    if n_range <= 1 or range_max_m <= 0:
        return 0.0
    return min(max(range_m / range_max_m * (n_range - 1), 0.0), float(n_range - 1))


def read_to_az_idx(az_deg: float, n_az: int, az_min_deg: float, az_max_deg: float) -> float:
    if n_az <= 1 or az_max_deg <= az_min_deg:
        return 0.0
    return min(max((az_deg - az_min_deg) / (az_max_deg - az_min_deg) * (n_az - 1), 0.0), float(n_az - 1))


def bbox_stats_from_ra(ra_db: np.ndarray, r0: int, r1: int, a0: int, a1: int) -> dict[str, Any]:
    r0 = max(0, min(r0, ra_db.shape[0] - 1))
    r1 = max(r0, min(r1, ra_db.shape[0] - 1))
    a0 = max(0, min(a0, ra_db.shape[1] - 1))
    a1 = max(a0, min(a1, ra_db.shape[1] - 1))
    patch = ra_db[r0:r1 + 1, a0:a1 + 1]
    vals = patch[np.isfinite(patch)]
    area_px = int(patch.size)
    peak_db = float(np.max(vals)) if vals.size else 0.0
    mean_db = float(np.mean(vals)) if vals.size else 0.0
    std_db = float(np.std(vals)) if vals.size else 0.0
    return {
        "range_start_idx": r0,
        "range_end_idx": r1,
        "az_start_idx": a0,
        "az_end_idx": a1,
        "centroid_range_idx": 0.5 * (r0 + r1),
        "centroid_az_idx": 0.5 * (a0 + a1),
        "area_px": area_px,
        "peak_db": peak_db,
        "mean_db": mean_db,
        "std_db": std_db,
        "fill_ratio": 1.0,
    }


def refine_bbox_to_energy(ra_db: np.ndarray, r0: int, r1: int, a0: int, a1: int) -> tuple[int, int, int, int]:
    n_range, n_az = ra_db.shape
    pred_cr = 0.5 * (r0 + r1)
    pred_ca = 0.5 * (a0 + a1)
    span_r = max(2, r1 - r0 + 1)
    span_a = max(2, a1 - a0 + 1)

    sr0 = max(0, r0 - max(6, span_r * 2))
    sr1 = min(n_range - 1, r1 + max(6, span_r * 2))
    sa0 = max(0, a0 - max(6, span_a * 3))
    sa1 = min(n_az - 1, a1 + max(6, span_a * 3))
    search = ra_db[sr0:sr1 + 1, sa0:sa1 + 1]
    vals = search[np.isfinite(search)]
    if vals.size == 0:
        return r0, r1, a0, a1

    thr = float(np.percentile(vals, 96.0))
    mask = np.isfinite(search) & (search >= thr)
    if not mask.any():
        peak_idx = np.unravel_index(int(np.nanargmax(search)), search.shape)
        pr = sr0 + int(peak_idx[0])
        pa = sa0 + int(peak_idx[1])
        hr = max(1, span_r // 2)
        ha = max(1, span_a // 2)
        return max(0, pr - hr), min(n_range - 1, pr + hr), max(0, pa - ha), min(n_az - 1, pa + ha)

    labels, nlab = ndi.label(mask)
    objs = ndi.find_objects(labels)
    best = None
    best_score = None
    for lab_id, slc in enumerate(objs, start=1):
        if slc is None:
            continue
        rr, cc = slc
        comp_mask = labels[rr, cc] == lab_id
        if not comp_mask.any():
            continue
        patch = search[rr, cc]
        comp_vals = patch[comp_mask]
        peak = float(np.max(comp_vals)) if comp_vals.size else -1e9
        cy, cx = ndi.center_of_mass(comp_mask)
        cr = sr0 + rr.start + float(cy)
        ca = sa0 + cc.start + float(cx)
        dist = math.hypot((cr - pred_cr) / max(span_r, 1), (ca - pred_ca) / max(span_a, 1))
        score = peak - 3.0 * dist
        if best_score is None or score > best_score:
            best_score = score
            best = (sr0 + rr.start, sr0 + rr.stop - 1, sa0 + cc.start, sa0 + cc.stop - 1)

    if best is None:
        return r0, r1, a0, a1

    br0, br1, ba0, ba1 = best
    if br1 - br0 + 1 < span_r:
        c = 0.5 * (br0 + br1)
        hr = max(1, span_r // 2)
        br0 = int(max(0, math.floor(c - hr)))
        br1 = int(min(n_range - 1, math.ceil(c + hr)))
    if ba1 - ba0 + 1 < span_a:
        c = 0.5 * (ba0 + ba1)
        ha = max(1, span_a // 2)
        ba0 = int(max(0, math.floor(c - ha)))
        ba1 = int(min(n_az - 1, math.ceil(c + ha)))
    return br0, br1, ba0, ba1


def _q(value: str, step: float) -> int:
    return int(round(safe_float(value, 0.0) / step))


def aggregate_powerline_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[dict[str, str]]] = defaultdict(list)
    for row_idx, row in enumerate(rows, start=1):
        x0 = safe_float(row.get("start_x_north_m"), 0.0)
        y0 = safe_float(row.get("start_y_west_m"), 0.0)
        z0 = safe_float(row.get("start_z_up_m"), 0.0)
        x1 = safe_float(row.get("end_x_north_m"), 0.0)
        y1 = safe_float(row.get("end_y_west_m"), 0.0)
        z1 = safe_float(row.get("end_z_up_m"), 0.0)
        ends = sorted([
            (_q(str(x0), 40.0), _q(str(y0), 40.0), _q(str(z0), 25.0)),
            (_q(str(x1), 40.0), _q(str(y1), 40.0), _q(str(z1), 25.0)),
        ])
        key = tuple(ends[0] + ends[1])
        groups[key].append(row)
    out = []
    for rows_g in groups.values():
        best = max(rows_g, key=lambda r: safe_int(r.get("frame_index"), -1))
        out.append({"row": best, "hits": len(rows_g)})
    return out


def aggregate_isolated_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    total_rows = len(rows)
    for row_idx, row in enumerate(rows, start=1):
        key = (
            _q(row.get("x_north_m", "0"), 35.0),
            _q(row.get("y_west_m", "0"), 35.0),
            _q(row.get("z_up_m", "0"), 25.0),
        )
        groups[key].append(row)
    out = []
    for rows_g in groups.values():
        best = max(rows_g, key=lambda r: safe_int(r.get("frame_index"), -1))
        out.append({"row": best, "hits": len(rows_g)})
    return out


def aggregate_dense_rows(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    per_poly: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        per_poly[(row.get("frame_index", ""), row.get("area_index", ""))].append(row)

    groups: dict[tuple[int, int, int, int], list[list[dict[str, str]]]] = defaultdict(list)
    for poly_rows in per_poly.values():
        xs = [safe_float(r.get("x_north_m"), 0.0) for r in poly_rows]
        ys = [safe_float(r.get("y_west_m"), 0.0) for r in poly_rows]
        zs = [safe_float(r.get("z_up_m"), 0.0) for r in poly_rows]
        key = (
            int(round(np.mean(xs) / 50.0)),
            int(round(np.mean(ys) / 50.0)),
            int(round(np.mean(zs) / 30.0)),
            len(poly_rows),
        )
        groups[key].append(poly_rows)

    out = []
    for polys in groups.values():
        best = max(polys, key=lambda rows_g: max(safe_int(r.get("frame_index"), -1) for r in rows_g))
        out.append(best)
    return out


def protocol_entries_from_scan_dir(scan_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    power_rows = load_csv_if_exists(scan_dir / "powerlines.csv")
    iso_rows = load_csv_if_exists(scan_dir / "isolated_objects.csv")
    dense_rows = load_csv_if_exists(scan_dir / "dense_areas.csv")

    power_aggr = aggregate_powerline_rows(power_rows)
    iso_aggr = aggregate_isolated_rows(iso_rows)
    dense_aggr = aggregate_dense_rows(dense_rows)

    for i, item in enumerate(power_aggr, start=1):
        entries.append({
            "source": "powerline",
            "component_id": i,
            "candidate_family": "powerline_segment",
            "weak_label": "Power line",
            "objectness": min(1.25, 0.55 + 0.08 * math.log1p(item["hits"])),
            "row": item["row"],
            "hits": item["hits"],
        })

    base = len(entries)
    for i, item in enumerate(iso_aggr, start=1):
        entries.append({
            "source": "isolated_object",
            "component_id": base + i,
            "candidate_family": "isolated_object",
            "weak_label": "Unknown",
            "objectness": min(1.10, 0.45 + 0.07 * math.log1p(item["hits"])),
            "row": item["row"],
            "hits": item["hits"],
        })

    base = len(entries)
    for i, rows_g in enumerate(dense_aggr, start=1):
        entries.append({
            "source": "dense_area",
            "component_id": base + i,
            "candidate_family": "dense_area",
            "weak_label": "Building complex",
            "objectness": min(1.05, 0.50 + 0.06 * math.log1p(len(rows_g))),
            "rows": rows_g,
            "hits": len(rows_g),
        })

    return entries


def load_terrain_points_for_scan(
    scan_dir: Path,
    scan_row: dict[str, str],
    ra_shape: tuple[int, ...],
) -> tuple[list[dict[str, float]], str, int]:
    terrain_csv = scan_dir / "terrain_points.csv"
    read_points_csv = scan_dir / "read_points.csv"
    rows = load_csv_if_exists(terrain_csv)
    source = "terrain_points.csv"
    raw_count = len(rows)
    if not rows and read_points_csv.exists():
        read_rows = load_csv_if_exists(read_points_csv)
        rows = [row for row in read_rows if row.get("source") == "terrain_point"]
        source = "read_points.csv:terrain_point"
        raw_count = len(rows)
    if not rows:
        return [], "missing", 0
    n_range, n_az = int(ra_shape[0]), int(ra_shape[1])
    range_max_m = safe_float(scan_row.get("range_max_m"), float(n_range))
    az_min_deg = safe_float(scan_row.get("az_min_deg"), -53.0)
    az_max_deg = safe_float(scan_row.get("az_max_deg"), 53.0)
    points: list[dict[str, float]] = []
    for row in rows:
        range_m = safe_float(row.get("range_gate_m", row.get("range_m")), float("nan"))
        azimuth_deg = safe_float(row.get("antenna_azimuth_deg", row.get("azimuth_deg")), float("nan"))
        elevation_deg = safe_float(row.get("antenna_pitch_deg", row.get("elevation_deg")), float("nan"))
        if not (math.isfinite(range_m) and math.isfinite(azimuth_deg) and math.isfinite(elevation_deg)):
            continue
        if range_m < 0.0 or range_m > range_max_m * 1.05:
            continue
        if azimuth_deg < az_min_deg - 3.0 or azimuth_deg > az_max_deg + 3.0:
            continue
        points.append({
            "range_m": range_m,
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "power": safe_float(row.get("power"), 0.0),
            "range_idx": read_to_range_idx(range_m, n_range, range_max_m),
            "az_idx": read_to_az_idx(azimuth_deg, n_az, az_min_deg, az_max_deg),
        })
    return points, source, raw_count


def ground_metrics_for_component(
    comp: dict[str, Any],
    scan_row: dict[str, str],
    terrain_points: list[dict[str, float]],
) -> dict[str, float]:
    az_span_deg = abs(safe_float(scan_row.get("az_max_deg"), 53.0) - safe_float(scan_row.get("az_min_deg"), -53.0))
    n_az = max(1, safe_int(str(comp.get("az_end_idx", 0)), 0) - safe_int(str(comp.get("az_start_idx", 0)), 0) + 1)
    az_bin_deg = az_span_deg / max(1.0, safe_float(scan_row.get("ra_shape_az_bins"), float("nan"))) if False else max(0.1, az_span_deg / 256.0)
    r0 = int(comp["range_start_idx"])
    r1 = int(comp["range_end_idx"])
    a0 = int(comp["az_start_idx"])
    a1 = int(comp["az_end_idx"])
    cr = float(comp["centroid_range_idx"])
    ca = float(comp["centroid_az_idx"])
    bbox_area = max(1, (r1 - r0 + 1) * (a1 - a0 + 1))
    pad_r = 2
    pad_a = 3
    in_count = 0
    near_count = 0
    nearest_dist = float("inf")
    nearest_clearance_m = float("inf")
    terrain_power_sum = 0.0
    terrain_power_hits = 0
    az_window_deg = max(1.5, abs(safe_float(comp.get("az_end_deg"), 0.0) - safe_float(comp.get("az_start_deg"), 0.0)) * 1.25)
    range_m = safe_float(comp.get("range_m"), float("nan"))
    elevation_deg = safe_float(comp.get("elevation_deg"), 0.0)
    for pt in terrain_points:
        pr = float(pt["range_idx"])
        pa = float(pt["az_idx"])
        if r0 <= pr <= r1 and a0 <= pa <= a1:
            in_count += 1
            terrain_power_sum += float(pt.get("power", 0.0))
            terrain_power_hits += 1
        if (r0 - pad_r) <= pr <= (r1 + pad_r) and (a0 - pad_a) <= pa <= (a1 + pad_a):
            near_count += 1
        dist = math.hypot((pr - cr) / max(2.0, (r1 - r0 + 1)), (pa - ca) / max(2.0, (a1 - a0 + 1)))
        nearest_dist = min(nearest_dist, dist)
        if abs(float(pt["azimuth_deg"]) - safe_float(comp.get("azimuth_deg"), 0.0)) <= az_window_deg:
            nearest_clearance_m = min(nearest_clearance_m, abs(range_m - float(pt["range_m"])))
    if not math.isfinite(nearest_dist):
        nearest_dist = 99.0
    if not math.isfinite(nearest_clearance_m):
        nearest_clearance_m = 1e6
    overlap_ratio = float(in_count / max(1, near_count)) if near_count > 0 else 0.0
    density_ratio = float(in_count / bbox_area)
    mean_power = terrain_power_sum / max(1, terrain_power_hits)
    range_span_m = safe_float(comp.get("range_span_m"), 0.0)
    az_span_deg = safe_float(comp.get("az_span_deg"), 0.0)
    fill_ratio = safe_float(comp.get("fill_ratio"), 0.0)
    shape_score = 0.0
    shape_score += 1.0 if abs(elevation_deg) <= 2.5 else 0.0
    shape_score += 1.0 if az_span_deg >= 3.0 else 0.0
    shape_score += 1.0 if range_span_m >= 60.0 else 0.0
    shape_score += 0.5 if fill_ratio >= 0.40 else 0.0

    near_ground_band_score = 0.0
    near_ground_band_score += 1.0 if range_m <= 900.0 else 0.0
    near_ground_band_score += 1.0 if abs(elevation_deg) <= 4.5 else 0.0
    near_ground_band_score += 1.0 if az_span_deg >= 2.5 else 0.0
    near_ground_band_score += 1.0 if range_span_m >= 18.0 else 0.0
    near_ground_band_score += 0.5 if fill_ratio >= 0.30 else 0.0
    near_ground_band_score += 0.5 if str(comp.get("candidate_family", "")) == "radar_blob" else 0.0
    return {
        "ground_point_count": float(in_count),
        "ground_near_point_count": float(near_count),
        "ground_overlap_ratio": overlap_ratio,
        "ground_density_ratio": density_ratio,
        "ground_nearest_norm_dist": nearest_dist,
        "ground_clearance_m": nearest_clearance_m,
        "ground_terrain_mean_power": mean_power,
        "ground_shape_score": shape_score,
        "near_ground_band_score": near_ground_band_score,
    }


def attach_ground_metrics(
    comp: dict[str, Any],
    scan_row: dict[str, str],
    terrain_points: list[dict[str, float]],
) -> dict[str, Any]:
    out = dict(comp)
    out.update(ground_metrics_for_component(out, scan_row, terrain_points))
    return out


def should_suppress_ground(comp: dict[str, Any]) -> bool:
    family = str(comp.get("candidate_family", ""))
    overlap = float(comp.get("ground_overlap_ratio", 0.0))
    near_count = float(comp.get("ground_near_point_count", 0.0))
    clearance = float(comp.get("ground_clearance_m", 1e6))
    nearest_dist = float(comp.get("ground_nearest_norm_dist", 99.0))
    shape_score = float(comp.get("ground_shape_score", 0.0))
    near_ground_band_score = float(comp.get("near_ground_band_score", 0.0))
    elevation_deg = abs(float(comp.get("elevation_deg", 0.0)))
    range_m = float(comp.get("range_m", 1e9))
    az_span_deg = float(comp.get("az_span_deg", 0.0))
    range_span_m = float(comp.get("range_span_m", 0.0))
    if family == "powerline_segment" and elevation_deg >= 4.0:
        return False
    if overlap >= 0.55 and near_count >= 2.0:
        return True
    if near_count >= 1.0 and clearance <= 18.0 and elevation_deg <= 2.5 and shape_score >= 1.5:
        return True
    if nearest_dist <= 0.65 and overlap >= 0.30 and shape_score >= 2.0:
        return True
    if family == "radar_blob" and range_m <= 900.0 and elevation_deg <= 4.5 and near_ground_band_score >= 3.5:
        return True
    if family == "radar_blob" and range_m <= 500.0 and elevation_deg <= 5.0 and (az_span_deg >= 2.0 or range_span_m >= 18.0):
        return True
    return False


def apply_ground_suppression(
    components: list[dict[str, Any]],
    scan_row: dict[str, str],
    terrain_points: list[dict[str, float]],
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    suppressed = 0
    for comp in components:
        c = attach_ground_metrics(comp, scan_row, terrain_points)
        if terrain_points and should_suppress_ground(c):
            suppressed += 1
            continue
        kept.append(c)
    if not terrain_points:
        kept = [attach_ground_metrics(comp, scan_row, terrain_points) for comp in components]
    return kept, suppressed


def protocol_entry_to_component(
    entry: dict[str, Any],
    scan_row: dict[str, str],
    ra_db: np.ndarray,
    ra_shape: tuple[int, ...],
) -> dict[str, Any] | None:
    n_range, n_az = int(ra_shape[0]), int(ra_shape[1])
    range_max_m = safe_float(scan_row.get("range_max_m"), float(n_range))
    az_min_deg = safe_float(scan_row.get("az_min_deg"), -53.0)
    az_max_deg = safe_float(scan_row.get("az_max_deg"), 53.0)
    min_range_bins = 2
    min_az_bins = 2

    if entry["source"] == "powerline":
        row = entry["row"]
        heading_deg = safe_float(row.get("aircraft_true_heading_deg"), float("nan"))
        p0 = xyz_nwu_to_body_read(safe_float(row.get("start_x_north_m"), 0.0), safe_float(row.get("start_y_west_m"), 0.0), safe_float(row.get("start_z_up_m"), 0.0), heading_deg)
        p1 = xyz_nwu_to_body_read(safe_float(row.get("end_x_north_m"), 0.0), safe_float(row.get("end_y_west_m"), 0.0), safe_float(row.get("end_z_up_m"), 0.0), heading_deg)
        pts = [p0, p1]
        source_row = row
    elif entry["source"] == "isolated_object":
        row = entry["row"]
        heading_deg = safe_float(row.get("aircraft_true_heading_deg"), float("nan"))
        pts = [xyz_nwu_to_body_read(safe_float(row.get("x_north_m"), 0.0), safe_float(row.get("y_west_m"), 0.0), safe_float(row.get("z_up_m"), 0.0), heading_deg)]
        source_row = row
    elif entry["source"] == "dense_area":
        pts = []
        for row in entry["rows"]:
            heading_deg = safe_float(row.get("aircraft_true_heading_deg"), float("nan"))
            pts.append(xyz_nwu_to_body_read(safe_float(row.get("x_north_m"), 0.0), safe_float(row.get("y_west_m"), 0.0), safe_float(row.get("z_up_m"), 0.0), heading_deg))
        if not pts:
            return None
        source_row = entry["rows"][0]
    else:
        return None

    ranges = [p["range_m"] for p in pts]
    azs = [p["azimuth_deg"] for p in pts]
    elevs = [p["elevation_deg"] for p in pts]

    az_margin_deg = 2.0
    range_margin_m = max(30.0, 0.03 * range_max_m)
    if max(azs) < az_min_deg - az_margin_deg or min(azs) > az_max_deg + az_margin_deg:
        return None
    if max(ranges) < -range_margin_m or min(ranges) > range_max_m + range_margin_m:
        return None

    r_idx = [read_to_range_idx(v, n_range, range_max_m) for v in ranges]
    a_idx = [read_to_az_idx(v, n_az, az_min_deg, az_max_deg) for v in azs]
    r0 = int(math.floor(min(r_idx)))
    r1 = int(math.ceil(max(r_idx)))
    a0 = int(math.floor(min(a_idx)))
    a1 = int(math.ceil(max(a_idx)))

    if r1 - r0 + 1 < min_range_bins:
        pad = max(0, (min_range_bins - (r1 - r0 + 1) + 1) // 2)
        r0 -= pad
        r1 += pad
    if a1 - a0 + 1 < min_az_bins:
        pad = max(0, (min_az_bins - (a1 - a0 + 1) + 1) // 2)
        a0 -= pad
        a1 += pad

    r0, r1, a0, a1 = refine_bbox_to_energy(ra_db, r0, r1, a0, a1)
    stats = bbox_stats_from_ra(ra_db, r0, r1, a0, a1)
    stats["component_id"] = int(entry["component_id"])
    stats["candidate_family"] = entry.get("candidate_family", "protocol_target")
    stats["weak_label"] = entry.get("weak_label", "Unknown")
    stats["objectness"] = float(entry.get("objectness", 0.8))
    stats["range_m"] = float(np.mean(ranges))
    stats["azimuth_deg"] = float(np.mean(azs))
    stats["elevation_deg"] = float(np.mean(elevs))
    stats["source_frame_index"] = safe_int(source_row.get("frame_index"), -1)
    stats["source_network_frame_no"] = safe_int(source_row.get("network_frame_no"), -1)
    return stats


def protocol_candidates_for_scan(
    scan_dir: Path,
    scan_row: dict[str, str],
    ra_db: np.ndarray,
    ra_shape: tuple[int, ...],
    max_candidates_per_scan: int,
) -> list[dict[str, Any]]:
    entries = protocol_entries_from_scan_dir(scan_dir)
    comps: list[dict[str, Any]] = []
    for entry in entries:
        comp = protocol_entry_to_component(entry, scan_row, ra_db, ra_shape)
        if comp is not None:
            comps.append(comp)
    comps = [c for c in comps if c["range_start_idx"] > 0 and c["az_start_idx"] > 0 and c["range_end_idx"] < ra_shape[0] - 1 and c["az_end_idx"] < ra_shape[1] - 1]
    comps.sort(key=lambda c: (float(c.get("objectness", 0.0)), float(c.get("peak_db", 0.0)), float(c.get("area_px", 0.0))), reverse=True)
    kept: list[dict[str, Any]] = []
    for comp in comps:
        if any(bbox_iou(comp, prev) >= 0.35 for prev in kept):
            continue
        kept.append(comp)
        if max_candidates_per_scan > 0 and len(kept) >= max_candidates_per_scan:
            break
    return kept


def detect_components(ra_db: np.ndarray, threshold_db: float, min_area_px: int) -> list[dict[str, Any]]:
    mask = np.isfinite(ra_db) & (ra_db >= threshold_db)
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    labels, _ = ndi.label(mask)
    boxes = ndi.find_objects(labels)

    comps: list[dict[str, Any]] = []
    for comp_id, slc in enumerate(boxes, start=1):
        if slc is None:
            continue
        rr, cc = slc
        comp_mask = labels[rr, cc] == comp_id
        area_px = int(comp_mask.sum())
        if area_px < min_area_px:
            continue
        patch = ra_db[rr, cc]
        masked_vals = patch[comp_mask]
        if masked_vals.size == 0:
            continue
        cy, cx = ndi.center_of_mass(comp_mask)
        comps.append({
            "component_id": comp_id,
            "range_start_idx": int(rr.start),
            "range_end_idx": int(rr.stop - 1),
            "az_start_idx": int(cc.start),
            "az_end_idx": int(cc.stop - 1),
            "centroid_range_idx": float(rr.start + cy),
            "centroid_az_idx": float(cc.start + cx),
            "area_px": area_px,
            "peak_db": float(np.max(masked_vals)),
            "mean_db": float(np.mean(masked_vals)),
            "std_db": float(np.std(masked_vals)),
            "fill_ratio": float(area_px / max(1, patch.size)),
            "candidate_family": "radar_blob",
            "weak_label": "Unknown",
            "objectness": 0.0,
        })
    return comps


def bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ar0, ar1 = a["range_start_idx"], a["range_end_idx"]
    aa0, aa1 = a["az_start_idx"], a["az_end_idx"]
    br0, br1 = b["range_start_idx"], b["range_end_idx"]
    ba0, ba1 = b["az_start_idx"], b["az_end_idx"]
    ir0, ir1 = max(ar0, br0), min(ar1, br1)
    ia0, ia1 = max(aa0, ba0), min(aa1, ba1)
    if ir1 < ir0 or ia1 < ia0:
        return 0.0
    inter = float((ir1 - ir0 + 1) * (ia1 - ia0 + 1))
    area_a = float((ar1 - ar0 + 1) * (aa1 - aa0 + 1))
    area_b = float((br1 - br0 + 1) * (ba1 - ba0 + 1))
    union = max(area_a + area_b - inter, 1.0)
    return inter / union


def component_score_key(comp: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(comp.get("objectness", 0.0)),
        float(comp.get("peak_db", 0.0)),
        float(comp.get("area_px", 0.0)),
    )


def merge_component_pair(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    use_secondary_bbox = component_score_key(secondary) > component_score_key(primary)
    if use_secondary_bbox:
        for key in [
            "range_start_idx", "range_end_idx", "az_start_idx", "az_end_idx",
            "centroid_range_idx", "centroid_az_idx", "area_px", "peak_db", "mean_db", "std_db", "fill_ratio",
        ]:
            merged[key] = secondary.get(key, merged.get(key))
    merged["objectness"] = max(float(primary.get("objectness", 0.0)), float(secondary.get("objectness", 0.0)))
    merged["candidate_source"] = "protocol+rae"
    merged["candidate_family"] = primary.get("candidate_family", secondary.get("candidate_family", "radar_blob"))
    merged["weak_label"] = primary.get("weak_label", secondary.get("weak_label", "Unknown"))
    for key in ["range_m", "azimuth_deg", "elevation_deg", "source_frame_index", "source_network_frame_no"]:
        if key in primary or key in secondary:
            merged[key] = primary.get(key, secondary.get(key))
    return merged


def fuse_component_lists(
    protocol_components: list[dict[str, Any]],
    rae_components: list[dict[str, Any]],
    max_candidates_per_scan: int,
) -> list[dict[str, Any]]:
    protocol_sorted = sorted(protocol_components, key=component_score_key, reverse=True)
    rae_sorted = sorted(rae_components, key=component_score_key, reverse=True)
    fused: list[dict[str, Any]] = []

    for comp in protocol_sorted:
        c = dict(comp)
        c.setdefault("candidate_source", "protocol")
        fused.append(c)

    for rae_comp in rae_sorted:
        rae_item = dict(rae_comp)
        rae_item.setdefault("candidate_source", "rae")
        matched = False
        for i, prev in enumerate(fused):
            if bbox_iou(rae_item, prev) >= 0.25:
                fused[i] = merge_component_pair(prev, rae_item)
                matched = True
                break
        if not matched:
            fused.append(rae_item)

    fused.sort(key=component_score_key, reverse=True)
    kept: list[dict[str, Any]] = []
    for comp in fused:
        if any(bbox_iou(comp, prev) >= 0.35 for prev in kept):
            continue
        kept.append(comp)
        if max_candidates_per_scan > 0 and len(kept) >= max(max_candidates_per_scan * 3, max_candidates_per_scan):
            break
    return kept


def range_band_id(range_m: float) -> int:
    if range_m < 800.0:
        return 0
    if range_m < 1600.0:
        return 1
    if range_m < 2600.0:
        return 2
    return 3


def balanced_range_quota(max_candidates_per_scan: int) -> list[int]:
    if max_candidates_per_scan <= 4:
        return [1, 1, 1, 1]
    q0 = max(2, int(round(max_candidates_per_scan * 0.34)))
    q1 = max(2, int(round(max_candidates_per_scan * 0.22)))
    q2 = max(2, int(round(max_candidates_per_scan * 0.22)))
    used = q0 + q1 + q2
    q3 = max(1, max_candidates_per_scan - used)
    quotas = [q0, q1, q2, q3]
    while sum(quotas) > max_candidates_per_scan:
        idx = max(range(4), key=lambda i: quotas[i])
        quotas[idx] -= 1
    while sum(quotas) < max_candidates_per_scan:
        quotas[-1] += 1
    return quotas


def select_components_balanced_by_range(
    components: list[dict[str, Any]],
    max_candidates_per_scan: int,
) -> list[dict[str, Any]]:
    if max_candidates_per_scan <= 0 or len(components) <= max_candidates_per_scan:
        return sorted(components, key=component_score_key, reverse=True)
    quotas = balanced_range_quota(max_candidates_per_scan)
    buckets: list[list[dict[str, Any]]] = [[], [], [], []]
    for comp in sorted(components, key=component_score_key, reverse=True):
        band = range_band_id(float(comp.get("range_m", 0.0)))
        buckets[band].append(comp)
    kept: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for band, quota in enumerate(quotas):
        band_items = buckets[band]
        kept.extend(band_items[:quota])
        leftovers.extend(band_items[quota:])
    if len(kept) < max_candidates_per_scan:
        leftovers.sort(key=component_score_key, reverse=True)
        for comp in leftovers:
            if len(kept) >= max_candidates_per_scan:
                break
            kept.append(comp)
    kept.sort(key=component_score_key, reverse=True)
    return kept[:max_candidates_per_scan]


def enrich_component_geometry(comp: dict[str, Any], scan_row: dict[str, str], ra_shape: tuple[int, ...]) -> dict[str, Any]:
    out = dict(comp)
    n_range, n_az = int(ra_shape[0]), int(ra_shape[1])
    range_max_m = safe_float(scan_row.get("range_max_m"), float(n_range))
    az_min_deg = safe_float(scan_row.get("az_min_deg"), -53.0)
    az_max_deg = safe_float(scan_row.get("az_max_deg"), 53.0)
    if not math.isfinite(float(out.get("range_m", float("nan")))) or not math.isfinite(float(out.get("azimuth_deg", float("nan")))):
        range_m, azimuth_deg = compute_range_azimuth(
            out["centroid_range_idx"],
            out["centroid_az_idx"],
            n_range,
            n_az,
            range_max_m,
            az_min_deg,
            az_max_deg,
        )
        out["range_m"] = range_m
        out["azimuth_deg"] = azimuth_deg
    range_start_m, _ = compute_range_azimuth(out["range_start_idx"], out["centroid_az_idx"], n_range, n_az, range_max_m, az_min_deg, az_max_deg)
    range_end_m, _ = compute_range_azimuth(out["range_end_idx"], out["centroid_az_idx"], n_range, n_az, range_max_m, az_min_deg, az_max_deg)
    _, az_start_deg = compute_range_azimuth(out["centroid_range_idx"], out["az_start_idx"], n_range, n_az, range_max_m, az_min_deg, az_max_deg)
    _, az_end_deg = compute_range_azimuth(out["centroid_range_idx"], out["az_end_idx"], n_range, n_az, range_max_m, az_min_deg, az_max_deg)
    out["range_start_m"] = range_start_m
    out["range_end_m"] = range_end_m
    out["az_start_deg"] = az_start_deg
    out["az_end_deg"] = az_end_deg
    out["range_span_m"] = abs(range_end_m - range_start_m)
    out["az_span_deg"] = abs(az_end_deg - az_start_deg)
    if not math.isfinite(float(out.get("elevation_deg", float("nan")))):
        out["elevation_deg"] = safe_float(scan_row.get("selected_pitch_deg"), 0.0)
    return out


def component_objectness(comp: dict[str, Any], threshold_db: float) -> float:
    area = float(comp["area_px"])
    peak = float(comp["peak_db"])
    mean = float(comp["mean_db"])
    fill = float(comp["fill_ratio"])
    az_span = float(comp["az_end_idx"] - comp["az_start_idx"] + 1)
    rg_span = float(comp["range_end_idx"] - comp["range_start_idx"] + 1)
    score = 0.0
    score += min(area / 120.0, 2.0)
    score += max(0.0, (peak - threshold_db) / 8.0)
    score += max(0.0, (mean - threshold_db + 3.0) / 6.0)
    score += 0.5 if 0.10 <= fill <= 0.85 else -0.3
    score += 0.4 if az_span >= 2 else -0.2
    score += 0.4 if rg_span >= 2 else -0.2
    return score


def prune_components(components: list[dict[str, Any]], threshold_db: float, ra_shape: tuple[int, ...], max_candidates_per_scan: int) -> list[dict[str, Any]]:
    if not components:
        return []
    n_range, n_az = int(ra_shape[0]), int(ra_shape[1])
    filtered = []
    for comp in components:
        if comp["range_start_idx"] <= 0 or comp["az_start_idx"] <= 0:
            continue
        if comp["range_end_idx"] >= n_range - 1 or comp["az_end_idx"] >= n_az - 1:
            continue
        obj = component_objectness(comp, threshold_db)
        if obj < 0.75:
            continue
        comp = dict(comp)
        comp["objectness"] = float(obj)
        filtered.append(comp)
    filtered.sort(key=lambda c: (c["objectness"], c["peak_db"], c["area_px"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for comp in filtered:
        if any(bbox_iou(comp, prev) >= 0.35 for prev in kept):
            continue
        kept.append(comp)
        if len(kept) >= max_candidates_per_scan:
            break
    return kept


def make_candidate(scan_row: dict[str, str], ra_shape: tuple[int, ...], comp: dict[str, Any], ra_path: Path) -> dict[str, Any]:
    n_range, n_az = int(ra_shape[0]), int(ra_shape[1])
    range_max_m = safe_float(scan_row.get("range_max_m"), float(n_range))
    az_min_deg = safe_float(scan_row.get("az_min_deg"), -53.0)
    az_max_deg = safe_float(scan_row.get("az_max_deg"), 53.0)

    if math.isfinite(float(comp.get("range_m", float("nan")))) and math.isfinite(float(comp.get("azimuth_deg", float("nan")))):
        range_m = float(comp.get("range_m", 0.0))
        azimuth_deg = float(comp.get("azimuth_deg", 0.0))
    else:
        range_m, azimuth_deg = compute_range_azimuth(
            comp["centroid_range_idx"],
            comp["centroid_az_idx"],
            n_range,
            n_az,
            range_max_m,
            az_min_deg,
            az_max_deg,
        )

    range_start_m, _ = compute_range_azimuth(comp["range_start_idx"], comp["centroid_az_idx"], n_range, n_az, range_max_m, az_min_deg, az_max_deg)
    range_end_m, _ = compute_range_azimuth(comp["range_end_idx"], comp["centroid_az_idx"], n_range, n_az, range_max_m, az_min_deg, az_max_deg)
    _, az_start_deg = compute_range_azimuth(comp["centroid_range_idx"], comp["az_start_idx"], n_range, n_az, range_max_m, az_min_deg, az_max_deg)
    _, az_end_deg = compute_range_azimuth(comp["centroid_range_idx"], comp["az_end_idx"], n_range, n_az, range_max_m, az_min_deg, az_max_deg)

    scan_index = safe_int(scan_row.get("scan_index"), 0)
    component_id = int(comp["component_id"])
    return {
        "candidate_id": f"scan{scan_index:04d}_{str(comp.get('candidate_family', 'blob'))}_{component_id:04d}",
        "scan_index": scan_index,
        "ra": scan_row.get("ra", ""),
        "ra_path": str(ra_path.resolve()),
        "pkt_start": safe_int(scan_row.get("pkt_start")),
        "pkt_end": safe_int(scan_row.get("pkt_end")),
        "scan_t_center": safe_float(scan_row.get("scan_t_center")),
        "candidate_family": str(comp.get("candidate_family", "radar_blob")),
        "weak_label": str(comp.get("weak_label", "Unknown")),
        "component_id": component_id,
        "source_frame_index": safe_int(comp.get("source_frame_index"), -1),
        "source_network_frame_no": safe_int(comp.get("source_network_frame_no"), -1),
        "range_m": range_m,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": safe_float(comp.get("elevation_deg"), safe_float(scan_row.get("selected_pitch_deg"), 0.0)),
        "range_max_m": range_max_m,
        "az_min_deg": az_min_deg,
        "az_max_deg": az_max_deg,
        "range_start_m": range_start_m,
        "range_end_m": range_end_m,
        "az_start_deg": az_start_deg,
        "az_end_deg": az_end_deg,
        "range_span_m": abs(range_end_m - range_start_m),
        "az_span_deg": abs(az_end_deg - az_start_deg),
        "area_px": int(comp["area_px"]),
        "peak_db": float(comp["peak_db"]),
        "mean_db": float(comp["mean_db"]),
        "std_db": float(comp["std_db"]),
        "fill_ratio": float(comp["fill_ratio"]),
        "range_start_idx": int(comp["range_start_idx"]),
        "range_end_idx": int(comp["range_end_idx"]),
        "az_start_idx": int(comp["az_start_idx"]),
        "az_end_idx": int(comp["az_end_idx"]),
        "centroid_range_idx": float(comp["centroid_range_idx"]),
        "centroid_az_idx": float(comp["centroid_az_idx"]),
        "ra_shape": "x".join(str(v) for v in ra_shape),
        "selected_pitch_deg": safe_float(scan_row.get("selected_pitch_deg")),
        "selected_pitch_idx": safe_int(scan_row.get("selected_pitch_idx")),
        "pitch_layer_count": safe_int(scan_row.get("pitch_layer_count"), 0),
        "scan_dir": safe_int(scan_row.get("scan_dir"), 0),
        "candidate_objectness": float(comp.get("objectness", 0.0)),
        "candidate_source": str(comp.get("candidate_source", "unknown")),
        "ground_point_count": float(comp.get("ground_point_count", 0.0)),
        "ground_near_point_count": float(comp.get("ground_near_point_count", 0.0)),
        "ground_overlap_ratio": float(comp.get("ground_overlap_ratio", 0.0)),
        "ground_density_ratio": float(comp.get("ground_density_ratio", 0.0)),
        "ground_nearest_norm_dist": float(comp.get("ground_nearest_norm_dist", 99.0)),
        "ground_clearance_m": float(comp.get("ground_clearance_m", 1e6)),
        "ground_terrain_mean_power": float(comp.get("ground_terrain_mean_power", 0.0)),
        "ground_shape_score": float(comp.get("ground_shape_score", 0.0)),
        "near_ground_band_score": float(comp.get("near_ground_band_score", 0.0)),
    }


def write_candidates_csv(out_path: Path, candidates: list[dict[str, Any]]) -> None:
    fieldnames = list(candidates[0].keys()) if candidates else [
        "candidate_id", "scan_index", "ra", "ra_path", "pkt_start", "pkt_end", "scan_t_center",
        "candidate_family", "weak_label", "component_id", "range_m", "azimuth_deg", "elevation_deg",
        "range_max_m", "az_min_deg", "az_max_deg", "range_start_m", "range_end_m", "az_start_deg", "az_end_deg",
        "range_span_m", "az_span_deg", "area_px", "peak_db", "mean_db", "std_db", "fill_ratio",
        "range_start_idx", "range_end_idx", "az_start_idx", "az_end_idx", "centroid_range_idx", "centroid_az_idx",
        "ra_shape", "selected_pitch_deg", "selected_pitch_idx", "pitch_layer_count", "scan_dir", "candidate_objectness", "candidate_source",
        "ground_point_count", "ground_near_point_count", "ground_overlap_ratio", "ground_density_ratio",
        "ground_nearest_norm_dist", "ground_clearance_m", "ground_terrain_mean_power", "ground_shape_score", "near_ground_band_score",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)


def main() -> None:
    args = parse_args()
    mapping_csv, ra_dir, out_dir = resolve_paths(args)
    cap_dir = args.cap_dir.resolve()

    rows_all = load_csv_rows(mapping_csv)
    protocol_scans_root = resolve_protocol_scans_root(cap_dir, args.protocol_root)
    protocol_scan_records = load_protocol_scan_records(protocol_scans_root) if args.candidate_source in {"fusion", "protocol"} else []
    rows = choose_best_row_per_scan(rows_all)
    if args.max_scans > 0:
        rows = rows[: args.max_scans]
    if not rows:
        raise RuntimeError(f"no aligned scan rows found in {mapping_csv}")

    candidates: list[dict[str, Any]] = []
    per_scan_stats: list[dict[str, Any]] = []
    missing_protocol_rows: list[dict[str, Any]] = []
    total_rows = len(rows)
    for row_idx, row in enumerate(rows, start=1):
        ra_name = row.get("ra", "")
        if not ra_name:
            continue
        ra_path = ra_dir / ra_name
        if not ra_path.exists():
            continue

        rae = np.load(ra_path, allow_pickle=False)
        if rae.ndim != 3 or rae.shape[2] == 0:
            continue
        layer_idx = choose_layer_index(row)
        layer_idx = min(max(layer_idx, 0), rae.shape[2] - 1)
        ra_db = power_to_db(rae[:, :, layer_idx])
        ra_smooth = ndi.gaussian_filter(ra_db, sigma=max(args.smooth_sigma, 0.0)) if args.smooth_sigma > 0 else ra_db

        finite_vals = ra_smooth[np.isfinite(ra_smooth)]
        if finite_vals.size == 0:
            continue
        threshold_db = float(args.min_peak_db) if args.min_peak_db is not None else float(np.percentile(finite_vals, args.threshold_percentile))

        protocol_components: list[dict[str, Any]] = []
        rae_components: list[dict[str, Any]] = []
        terrain_points: list[dict[str, float]] = []
        terrain_source = "missing"
        terrain_raw_count = 0
        suppressed_ground_count = 0

        if args.candidate_source in {"fusion", "protocol"}:
            scan_dir = find_protocol_scan_dir(protocol_scan_records, row)
            if scan_dir is None:
                missing_protocol_rows.append({
                    "scan_index": safe_int(row.get("scan_index"), -1),
                    "pkt_start": safe_int(row.get("pkt_start"), -1),
                    "pkt_end": safe_int(row.get("pkt_end"), -1),
                    "ra": ra_name,
                })
                if args.candidate_source == "protocol":
                    continue
            else:
                terrain_points, terrain_source, terrain_raw_count = load_terrain_points_for_scan(scan_dir, row, tuple(int(v) for v in rae.shape))
                protocol_components = protocol_candidates_for_scan(
                    scan_dir=scan_dir,
                    scan_row=row,
                    ra_db=ra_smooth,
                    ra_shape=tuple(int(v) for v in rae.shape),
                    max_candidates_per_scan=max(args.max_candidates_per_scan * 3, args.max_candidates_per_scan),
                )
                for comp in protocol_components:
                    comp["candidate_source"] = "protocol"

        if args.candidate_source in {"fusion", "rae"}:
            rae_components = detect_components(ra_smooth, threshold_db, args.min_area_px)
            rae_components = prune_components(
                rae_components,
                threshold_db,
                tuple(int(v) for v in rae.shape),
                max(args.max_candidates_per_scan * 3, args.max_candidates_per_scan),
            )
            for comp in rae_components:
                comp["candidate_source"] = "rae"

        if args.candidate_source == "protocol":
            components = protocol_components
        elif args.candidate_source == "rae":
            components = rae_components
        else:
            components = fuse_component_lists(protocol_components, rae_components, args.max_candidates_per_scan)

        components = [enrich_component_geometry(comp, row, tuple(int(v) for v in rae.shape)) for comp in components]
        components, suppressed_ground_count = apply_ground_suppression(components, row, terrain_points)
        components = select_components_balanced_by_range(components, args.max_candidates_per_scan)

        for comp in components:
            candidates.append(make_candidate(row, tuple(int(v) for v in rae.shape), comp, ra_path))

        per_scan_stats.append({
            "scan_index": safe_int(row.get("scan_index")),
            "ra": ra_name,
            "layer_idx": layer_idx,
            "threshold_db": threshold_db,
            "component_count": len(components),
            "candidate_source": args.candidate_source,
            "terrain_point_count": len(terrain_points),
            "terrain_source": terrain_source,
            "terrain_raw_count": terrain_raw_count,
            "suppressed_ground_count": suppressed_ground_count,
            "range_band_quota": balanced_range_quota(args.max_candidates_per_scan),
        })

        if row_idx == 1 or row_idx % 5 == 0 or row_idx == total_rows:
            print(f"[SCAN] {row_idx}/{total_rows} scan_index={safe_int(row.get('scan_index'), -1)} candidates={len(components)} terrain={len(terrain_points)}/{terrain_raw_count} terrain_src={terrain_source} suppressed_ground={suppressed_ground_count} source={args.candidate_source}")

        if args.save_preview_json:
            preview_path = out_dir / "preview" / f"scan_{safe_int(row.get('scan_index')):06d}.json"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview_path.write_text(json.dumps({"row": row, "stats": per_scan_stats[-1], "components": components}, ensure_ascii=False, indent=2), encoding="utf-8")

    candidates.sort(key=lambda r: (int(r["scan_index"]), int(r["component_id"])))

    if args.candidate_source in {"fusion", "protocol"} and missing_protocol_rows:
        print(f"[WARN] skipped {len(missing_protocol_rows)} aligned scans because protocol scan rows were missing under {protocol_scans_root}")
        for item in missing_protocol_rows[:8]:
            print(f"  [WARN] missing protocol scan for aligned scan_index={item['scan_index']} pkt={item['pkt_start']}..{item['pkt_end']} ra={item['ra']}")

    if args.candidate_source in {"fusion", "protocol"} and not protocol_scan_records:
        raise RuntimeError(
            f"no protocol-backed scans could be matched under {protocol_scans_root}. "
            f"Please rerun parse_mmwave_bin_with_read.py on the full bin, or use --candidate-source rae."
        )

    if not candidates:
        raise RuntimeError("no radar candidates generated")

    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = out_dir / "candidates.csv"
    manifest_path = out_dir / "candidates_manifest.json"
    write_candidates_csv(candidate_csv, candidates)
    manifest = {
        "cap_dir": str(cap_dir),
        "mapping_csv": str(mapping_csv),
        "ra_dir": str(ra_dir),
        "protocol_scans_root": str(protocol_scans_root),
        "candidate_source": args.candidate_source,
        "selected_scans": len(rows),
        "matched_scans": len(per_scan_stats),
        "candidates": len(candidates),
        "missing_protocol_scans": len(missing_protocol_rows),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"capture dir: {cap_dir}")
    print(f"candidate out dir: {out_dir}")
    print(f"mapping csv: {mapping_csv}")
    print(f"ra dir: {ra_dir}")
    print(f"selected scans: {len(rows)}")
    print(f"matched scans: {len(per_scan_stats)}")
    print(f"candidates: {len(candidates)}")
    print(f"candidate csv: {candidate_csv}")
    print(f"manifest json: {manifest_path}")


if __name__ == "__main__":
    main()
