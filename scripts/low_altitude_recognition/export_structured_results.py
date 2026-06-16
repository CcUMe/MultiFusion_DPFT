from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import load_csv_rows, safe_float, safe_int

CAM_STABLE_SUBDIR = "hikrobot_camera__DA8679037__image_raw_stablelized_1920x1200"
CAM_RAW_SUBDIR = "hikrobot_camera__DA8679037__image_raw"


NUMERIC_KEYS = {
    "camera_t_bag", "scan_t_start", "scan_t_end", "scan_t_center", "selected_pitch_deg",
    "selected_pitch_idx", "pitch_layer_count", "scan_dir", "az_min_deg", "az_max_deg", "range_max_m",
    "pred_confidence", "pred_objectness", "pred_groundness", "x_forward_m", "y_left_m", "z_up_m",
    "geom_range_m", "geom_azimuth_deg", "geom_elevation_deg", "geom_range_span_m", "geom_az_span_deg",
    "geom_area_px", "geom_fill_ratio", "geom_peak_db", "geom_mean_db", "geom_std_db",
    "candidate_objectness", "range_m", "azimuth_deg", "elevation_deg", "range_start_m", "range_end_m",
    "az_start_deg", "az_end_deg", "range_span_m", "az_span_deg", "area_px", "peak_db", "mean_db", "std_db",
    "fill_ratio", "ground_point_count", "ground_near_point_count", "ground_overlap_ratio", "ground_density_ratio",
    "ground_nearest_norm_dist", "ground_clearance_m", "ground_terrain_mean_power", "ground_shape_score",
    "near_ground_band_score", "source_frame_index", "source_network_frame_no", "pkt_start", "pkt_end",
    "centroid_range_idx", "centroid_az_idx", "range_start_idx", "range_end_idx", "az_start_idx", "az_end_idx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export structured low-altitude radar recognition results with image mappings.")
    parser.add_argument("--prediction-csv", type=Path, required=True, help="Prediction CSV produced by rule_based_recognition.py")
    parser.add_argument("--cap-dir", type=Path, required=True, help="Capture directory with aligned mapping CSV and images")
    parser.add_argument("--mapping-csv", type=Path, help="Aligned mapping CSV. Defaults to <cap-dir>/image_to_antframe_time_aligned.csv")
    parser.add_argument("--overlay-dir", type=Path, help="Radar overlay directory. Defaults to <prediction-dir>/radar_overlays")
    parser.add_argument("--out-dir", type=Path, help="Output structured result directory. Defaults to <prediction-dir>/structured_results")
    return parser.parse_args()


def to_number_if_possible(key: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if key not in NUMERIC_KEYS:
        return value
    text = str(value).strip()
    if text == "":
        return None
    iv = safe_int(text, default=10**18)
    fv = safe_float(text, default=float("nan"))
    if fv == fv:
        if abs(fv - round(fv)) < 1e-9 and abs(fv) < 9e15:
            return int(round(fv))
        return float(fv)
    if iv != 10**18:
        return iv
    return value


def convert_row(row: dict[str, str]) -> dict[str, Any]:
    return {k: to_number_if_possible(k, v) for k, v in row.items()}


def find_image_path(cap_dir: Path, row: dict[str, str]) -> Path | None:
    part_name = row.get("part_name", "")
    image_name = row.get("image", "")
    if not part_name or not image_name:
        return None
    part_dir = cap_dir / part_name
    candidates = [
        part_dir / "images" / CAM_STABLE_SUBDIR / image_name,
        part_dir / "images" / CAM_RAW_SUBDIR / image_name,
    ]
    if part_dir.exists():
        for seg_dir in sorted(part_dir.iterdir()):
            if not seg_dir.is_dir() or not seg_dir.name.startswith("segment_"):
                continue
            candidates.extend([
                seg_dir / "images" / CAM_STABLE_SUBDIR / image_name,
                seg_dir / "images" / CAM_RAW_SUBDIR / image_name,
            ])
    for path in candidates:
        if path.exists():
            return path
    return None


def unique_mapping_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (row.get("ra", ""), row.get("part_name", ""), row.get("image", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")



def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pred_csv = args.prediction_csv.resolve()
    cap_dir = args.cap_dir.resolve()
    mapping_csv = args.mapping_csv.resolve() if args.mapping_csv else cap_dir / "image_to_antframe_time_aligned.csv"
    overlay_dir = args.overlay_dir.resolve() if args.overlay_dir else pred_csv.parent / "radar_overlays"
    out_dir = args.out_dir.resolve() if args.out_dir else pred_csv.parent / "structured_results"

    pred_rows_raw = load_csv_rows(pred_csv)
    if not pred_rows_raw:
        raise RuntimeError(f"no prediction rows found in {pred_csv}")
    pred_rows = [convert_row(r) for r in pred_rows_raw]

    mapping_rows_raw = load_csv_rows(mapping_csv) if mapping_csv.exists() else []
    mapping_rows_raw = unique_mapping_rows(mapping_rows_raw)
    mapping_rows = [convert_row(r) for r in mapping_rows_raw]

    scan_to_dets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ra_to_dets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        scan_to_dets[str(row.get("scan_index", ""))].append(row)
        ra_to_dets[str(row.get("ra", ""))].append(row)

    ra_to_images: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_raw, row in zip(mapping_rows_raw, mapping_rows):
        ra = str(row.get("ra", ""))
        image_path = find_image_path(cap_dir, row_raw)
        row["resolved_image_path"] = str(image_path.resolve()) if image_path is not None else None
        row["resolved_image_exists"] = bool(image_path is not None)
        ra_to_images[ra].append(row)

    scan_records: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    image_detection_rows: list[dict[str, Any]] = []
    image_summary_rows: list[dict[str, Any]] = []

    for ra, detections in sorted(ra_to_dets.items(), key=lambda kv: (safe_int(kv[1][0].get("scan_index"), 0), kv[0])):
        first = detections[0]
        scan_index = safe_int(first.get("scan_index"), 0)
        overlay_path = overlay_dir / f"{Path(ra).stem}.png"
        matched_images = ra_to_images.get(ra, [])

        scan_record = {
            "scan_index": scan_index,
            "ra": ra,
            "ra_path": first.get("ra_path"),
            "pkt_start": first.get("pkt_start"),
            "pkt_end": first.get("pkt_end"),
            "scan_t_center": first.get("scan_t_center"),
            "part_names": sorted({str(img.get("part_name")) for img in matched_images if img.get("part_name")}),
            "matched_image_count": len(matched_images),
            "overlay_path": str(overlay_path.resolve()) if overlay_path.exists() else None,
            "detections": detections,
            "images": matched_images,
        }
        scan_records.append(scan_record)

        for det in detections:
            det_row = dict(det)
            det_row["matched_image_count"] = len(matched_images)
            det_row["overlay_path"] = str(overlay_path.resolve()) if overlay_path.exists() else None
            detection_rows.append(det_row)

        for img in matched_images:
            image_record = {
                "image": img.get("image"),
                "image_path": img.get("resolved_image_path"),
                "image_exists": img.get("resolved_image_exists"),
                "part_name": img.get("part_name"),
                "camera_seq": img.get("camera_seq"),
                "camera_t_bag": img.get("camera_t_bag"),
                "scan_index": scan_index,
                "ra": ra,
                "pkt_start": first.get("pkt_start"),
                "pkt_end": first.get("pkt_end"),
                "scan_t_center": first.get("scan_t_center"),
                "overlay_path": str(overlay_path.resolve()) if overlay_path.exists() else None,
                "detections": detections,
            }
            image_records.append(image_record)
            image_summary_rows.append({
                "image": img.get("image"),
                "image_path": img.get("resolved_image_path"),
                "image_exists": img.get("resolved_image_exists"),
                "part_name": img.get("part_name"),
                "camera_seq": img.get("camera_seq"),
                "camera_t_bag": img.get("camera_t_bag"),
                "scan_index": scan_index,
                "ra": ra,
                "overlay_path": str(overlay_path.resolve()) if overlay_path.exists() else None,
                "detection_count": len(detections),
            })
            for det in detections:
                image_detection_rows.append({
                    "image": img.get("image"),
                    "image_path": img.get("resolved_image_path"),
                    "image_exists": img.get("resolved_image_exists"),
                    "part_name": img.get("part_name"),
                    "camera_seq": img.get("camera_seq"),
                    "camera_t_bag": img.get("camera_t_bag"),
                    "scan_index": scan_index,
                    "ra": ra,
                    "overlay_path": str(overlay_path.resolve()) if overlay_path.exists() else None,
                    "candidate_id": det.get("candidate_id"),
                    "pred_label": det.get("pred_label"),
                    "pred_confidence": det.get("pred_confidence"),
                    "pred_groundness": det.get("pred_groundness"),
                    "candidate_family": det.get("candidate_family"),
                    "range_m": det.get("geom_range_m", det.get("range_m")),
                    "azimuth_deg": det.get("geom_azimuth_deg", det.get("azimuth_deg")),
                    "elevation_deg": det.get("geom_elevation_deg", det.get("elevation_deg")),
                    "x_forward_m": det.get("x_forward_m"),
                    "y_left_m": det.get("y_left_m"),
                    "z_up_m": det.get("z_up_m"),
                })

    manifest = {
        "cap_dir": str(cap_dir),
        "prediction_csv": str(pred_csv),
        "mapping_csv": str(mapping_csv),
        "overlay_dir": str(overlay_dir),
        "scan_count": len(scan_records),
        "image_count": len(image_records),
        "detection_count": len(pred_rows),
        "files": {
            "scans_jsonl": str((out_dir / 'scan_results.jsonl').resolve()),
            "images_jsonl": str((out_dir / 'image_results.jsonl').resolve()),
            "detections_csv": str((out_dir / 'detections.csv').resolve()),
            "image_detections_csv": str((out_dir / 'image_detections.csv').resolve()),
            "image_summary_csv": str((out_dir / 'images.csv').resolve()),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / 'scan_results.jsonl', scan_records)
    write_jsonl(out_dir / 'image_results.jsonl', image_records)
    write_csv(out_dir / 'detections.csv', detection_rows)
    write_csv(out_dir / 'image_detections.csv', image_detection_rows)
    write_csv(out_dir / 'images.csv', image_summary_rows)
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'prediction csv: {pred_csv}')
    print(f'structured result dir: {out_dir}')
    print(f'scan jsonl: {out_dir / "scan_results.jsonl"}')
    print(f'image jsonl: {out_dir / "image_results.jsonl"}')
    print(f'detections csv: {out_dir / "detections.csv"}')
    print(f'image detections csv: {out_dir / "image_detections.csv"}')
    print(f'images csv: {out_dir / "images.csv"}')
    print(f'manifest json: {out_dir / "manifest.json"}')


if __name__ == "__main__":
    main()
