from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import load_csv_rows, power_to_db, safe_float, safe_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract handcrafted radar-only features for low-altitude traditional recognition.")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="Candidate directory produced by build_candidates.py")
    parser.add_argument("--out-csv", type=Path, help="Output feature CSV path. Defaults to <candidate-dir>/features.csv")
    return parser.parse_args()


def crop_radar_patch(rae: np.ndarray, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    r0 = max(0, safe_int(row.get("range_start_idx"), 0))
    r1 = min(rae.shape[0], safe_int(row.get("range_end_idx"), rae.shape[0] - 1) + 1)
    a0 = max(0, safe_int(row.get("az_start_idx"), 0))
    a1 = min(rae.shape[1], safe_int(row.get("az_end_idx"), rae.shape[1] - 1) + 1)
    full_patch = rae[r0:r1, a0:a1, :]
    layer_idx = min(max(safe_int(row.get("selected_pitch_idx"), 0), 0), max(0, rae.shape[2] - 1))
    selected_patch = full_patch[:, :, layer_idx] if full_patch.size else np.zeros((0, 0), dtype=np.float32)
    return full_patch, selected_patch


def normalized_entropy(x: np.ndarray) -> float:
    vals = np.asarray(x, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size <= 1:
        return 0.0
    p = vals / np.sum(vals)
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(ent / math.log(len(p) + 1e-12))


def half_power_width(profile: np.ndarray) -> float:
    vals = np.asarray(profile, dtype=np.float64)
    vals = vals[np.isfinite(vals)] if vals.ndim == 1 else vals.ravel()
    if vals.size == 0:
        return 0.0
    peak = float(np.max(vals))
    if peak <= 0:
        return 0.0
    return float(np.sum(vals >= 0.5 * peak))


def radar_features(rae: np.ndarray, row: dict[str, str]) -> dict[str, float]:
    full_patch, selected_patch = crop_radar_patch(rae, row)
    selected_db = power_to_db(selected_patch) if selected_patch.size else np.zeros((0, 0), dtype=np.float32)

    feats: dict[str, float] = {}
    vals = selected_db[np.isfinite(selected_db)]
    feats["radar_patch_mean_db"] = float(np.mean(vals)) if vals.size else 0.0
    feats["radar_patch_std_db"] = float(np.std(vals)) if vals.size else 0.0
    feats["radar_patch_max_db"] = float(np.max(vals)) if vals.size else 0.0
    feats["radar_patch_p75_db"] = float(np.percentile(vals, 75.0)) if vals.size else 0.0
    feats["radar_patch_p90_db"] = float(np.percentile(vals, 90.0)) if vals.size else 0.0
    feats["radar_patch_p99_db"] = float(np.percentile(vals, 99.0)) if vals.size else 0.0
    feats["radar_patch_height_px"] = float(selected_patch.shape[0])
    feats["radar_patch_width_px"] = float(selected_patch.shape[1])
    feats["radar_patch_aspect"] = float(selected_patch.shape[1] / max(1, selected_patch.shape[0])) if selected_patch.size else 0.0

    if selected_patch.size:
        amp = np.abs(selected_patch).astype(np.float64)
        range_profile = np.mean(amp, axis=1)
        az_profile = np.mean(amp, axis=0)
        feats["radar_range_profile_peak_ratio"] = float(np.max(range_profile) / max(1e-6, np.sum(range_profile)))
        feats["radar_az_profile_peak_ratio"] = float(np.max(az_profile) / max(1e-6, np.sum(az_profile)))
        feats["radar_range_profile_entropy"] = normalized_entropy(range_profile)
        feats["radar_az_profile_entropy"] = normalized_entropy(az_profile)
        feats["radar_range_profile_hpw"] = half_power_width(range_profile)
        feats["radar_az_profile_hpw"] = half_power_width(az_profile)
        grad_r, grad_a = np.gradient(selected_db)
        grad_mag = np.hypot(grad_r, grad_a)
        feats["radar_grad_range_mean"] = float(np.mean(np.abs(grad_r)))
        feats["radar_grad_az_mean"] = float(np.mean(np.abs(grad_a)))
        feats["radar_grad_mag_mean"] = float(np.mean(np.abs(grad_mag)))
        thr = float(np.percentile(vals, 80.0)) if vals.size else 0.0
        mask = selected_db >= thr
        feats["radar_hot_fraction"] = float(np.mean(mask))
    else:
        feats["radar_range_profile_peak_ratio"] = 0.0
        feats["radar_az_profile_peak_ratio"] = 0.0
        feats["radar_range_profile_entropy"] = 0.0
        feats["radar_az_profile_entropy"] = 0.0
        feats["radar_range_profile_hpw"] = 0.0
        feats["radar_az_profile_hpw"] = 0.0
        feats["radar_grad_range_mean"] = 0.0
        feats["radar_grad_az_mean"] = 0.0
        feats["radar_grad_mag_mean"] = 0.0
        feats["radar_hot_fraction"] = 0.0

    if full_patch.size:
        layer_energy = np.mean(np.abs(full_patch), axis=(0, 1))
        feats["radar_layer_energy_mean"] = float(np.mean(layer_energy))
        feats["radar_layer_energy_std"] = float(np.std(layer_energy))
        feats["radar_layer_energy_max"] = float(np.max(layer_energy))
        feats["radar_layer_energy_entropy"] = normalized_entropy(layer_energy)
        feats["radar_active_layer_count"] = float(np.sum(layer_energy >= max(np.max(layer_energy) * 0.5, 1e-6)))
        feats["radar_layer_halfmax_width"] = half_power_width(layer_energy)
        sel_idx = min(max(safe_int(row.get("selected_pitch_idx"), 0), 0), len(layer_energy) - 1)
        feats["radar_selected_layer_energy"] = float(layer_energy[sel_idx])
        feats["radar_selected_layer_energy_ratio"] = float(layer_energy[sel_idx] / max(1e-6, np.sum(layer_energy)))
    else:
        feats["radar_layer_energy_mean"] = 0.0
        feats["radar_layer_energy_std"] = 0.0
        feats["radar_layer_energy_max"] = 0.0
        feats["radar_layer_energy_entropy"] = 0.0
        feats["radar_active_layer_count"] = 0.0
        feats["radar_layer_halfmax_width"] = 0.0
        feats["radar_selected_layer_energy"] = 0.0
        feats["radar_selected_layer_energy_ratio"] = 0.0

    return feats


def geometry_features(row: dict[str, str]) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats["geom_range_m"] = safe_float(row.get("range_m"), 0.0)
    feats["geom_azimuth_deg"] = safe_float(row.get("azimuth_deg"), 0.0)
    feats["geom_elevation_deg"] = safe_float(row.get("elevation_deg", row.get("selected_pitch_deg")), 0.0)
    feats["geom_range_span_m"] = safe_float(row.get("range_span_m"), 0.0)
    feats["geom_az_span_deg"] = safe_float(row.get("az_span_deg"), 0.0)
    feats["geom_area_px"] = safe_float(row.get("area_px"), 0.0)
    feats["geom_fill_ratio"] = safe_float(row.get("fill_ratio"), 0.0)
    feats["geom_peak_db"] = safe_float(row.get("peak_db"), 0.0)
    feats["geom_mean_db"] = safe_float(row.get("mean_db"), 0.0)
    feats["geom_std_db"] = safe_float(row.get("std_db"), 0.0)
    feats["geom_pitch_deg"] = safe_float(row.get("selected_pitch_deg"), 0.0)
    feats["geom_pitch_idx"] = float(safe_int(row.get("selected_pitch_idx"), 0))
    feats["geom_pitch_count"] = float(safe_int(row.get("pitch_layer_count"), 0))
    feats["geom_scan_dir"] = float(safe_int(row.get("scan_dir"), 0))
    feats["geom_abs_azimuth_deg"] = abs(feats["geom_azimuth_deg"])
    feats["geom_slant_area"] = feats["geom_range_span_m"] * max(feats["geom_az_span_deg"], 1e-6)
    feats["ground_point_count"] = safe_float(row.get("ground_point_count"), 0.0)
    feats["ground_near_point_count"] = safe_float(row.get("ground_near_point_count"), 0.0)
    feats["ground_overlap_ratio"] = safe_float(row.get("ground_overlap_ratio"), 0.0)
    feats["ground_density_ratio"] = safe_float(row.get("ground_density_ratio"), 0.0)
    feats["ground_nearest_norm_dist"] = safe_float(row.get("ground_nearest_norm_dist"), 99.0)
    feats["ground_clearance_m"] = safe_float(row.get("ground_clearance_m"), 1e6)
    feats["ground_terrain_mean_power"] = safe_float(row.get("ground_terrain_mean_power"), 0.0)
    feats["ground_shape_score"] = safe_float(row.get("ground_shape_score"), 0.0)
    feats["near_ground_band_score"] = safe_float(row.get("near_ground_band_score"), 0.0)
    feats["near_ground_flag"] = 1.0 if (
        feats["ground_overlap_ratio"] >= 0.30
        or (feats["ground_clearance_m"] <= 20.0 and abs(feats["geom_elevation_deg"]) <= 2.5)
        or feats["near_ground_band_score"] >= 3.5
    ) else 0.0
    return feats


def family_flags(row: dict[str, str]) -> dict[str, float]:
    family = row.get("candidate_family", "")
    return {
        "fam_powerline": 1.0 if family == "powerline_segment" else 0.0,
        "fam_isolated": 1.0 if family == "isolated_object" else 0.0,
        "fam_dense": 1.0 if family == "dense_area" else 0.0,
        "fam_blob": 1.0 if family == "radar_blob" else 0.0,
    }


def write_feature_csv(out_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("no feature rows extracted")
    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    candidate_dir = args.candidate_dir.resolve()
    candidate_csv = candidate_dir / "candidates.csv"
    out_csv = args.out_csv.resolve() if args.out_csv else candidate_dir / "features.csv"

    rows = load_csv_rows(candidate_csv)
    if not rows:
        raise RuntimeError(f"no candidate rows found in {candidate_csv}")

    feature_rows: list[dict[str, Any]] = []
    ra_cache: dict[str, np.ndarray] = {}
    for row in rows:
        ra_path_text = row.get("ra_path", "")
        ra_path = Path(ra_path_text) if ra_path_text else Path(row["ra"])
        if not ra_path.is_absolute():
            ra_path = candidate_dir.parent / "mmwave_ra_npy" / row["ra"]
        if str(ra_path) not in ra_cache:
            ra_cache[str(ra_path)] = np.load(ra_path, allow_pickle=False)
        rae = ra_cache[str(ra_path)]

        feat_row: dict[str, Any] = {
            "candidate_id": row.get("candidate_id", ""),
            "scan_index": row.get("scan_index", ""),
            "ra": row.get("ra", ""),
            "ra_path": row.get("ra_path", ""),
            "candidate_family": row.get("candidate_family", ""),
            "weak_label": row.get("weak_label", ""),
            "component_id": row.get("component_id", ""),
            "candidate_objectness": row.get("candidate_objectness", ""),
            "range_start_idx": row.get("range_start_idx", ""),
            "range_end_idx": row.get("range_end_idx", ""),
            "az_start_idx": row.get("az_start_idx", ""),
            "az_end_idx": row.get("az_end_idx", ""),
            "centroid_range_idx": row.get("centroid_range_idx", ""),
            "centroid_az_idx": row.get("centroid_az_idx", ""),
            "range_max_m": row.get("range_max_m", ""),
            "az_min_deg": row.get("az_min_deg", ""),
            "az_max_deg": row.get("az_max_deg", ""),
            "range_start_m": row.get("range_start_m", ""),
            "range_end_m": row.get("range_end_m", ""),
            "az_start_deg": row.get("az_start_deg", ""),
            "az_end_deg": row.get("az_end_deg", ""),
            "selected_pitch_deg": row.get("selected_pitch_deg", ""),
            "selected_pitch_idx": row.get("selected_pitch_idx", ""),
            "pitch_layer_count": row.get("pitch_layer_count", ""),
            "scan_dir": row.get("scan_dir", ""),
        }
        feat_row.update(geometry_features(row))
        feat_row.update(radar_features(rae, row))
        feat_row.update(family_flags(row))
        if "label" in row:
            feat_row["label"] = row["label"]
        if "split" in row:
            feat_row["split"] = row["split"]
        feature_rows.append(feat_row)

    write_feature_csv(out_csv, feature_rows)
    print(f"candidate dir: {candidate_dir}")
    print(f"candidate csv: {candidate_csv}")
    print(f"feature csv: {out_csv}")
    print(f"feature rows: {len(feature_rows)}")


if __name__ == "__main__":
    main()
