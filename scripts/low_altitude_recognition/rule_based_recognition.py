from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import TARGET_CLASSES, load_csv_rows, safe_float, safe_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based 8-class recognition using radar-only handcrafted features.")
    parser.add_argument("--feature-csv", type=Path, required=True, help="Feature CSV produced by extract_features.py")
    parser.add_argument("--out-csv", type=Path, help="Prediction CSV path. Defaults to <feature-dir>/predictions.csv")
    return parser.parse_args()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def softmax(scores: dict[str, float]) -> dict[str, float]:
    vals = np.asarray(list(scores.values()), dtype=np.float64)
    vals = vals - np.max(vals)
    expv = np.exp(vals)
    probs = expv / np.sum(expv)
    return {cls: float(p) for cls, p in zip(scores.keys(), probs)}


def groundness(row: dict[str, str]) -> float:
    overlap = safe_float(row.get("ground_overlap_ratio"), 0.0)
    clearance = safe_float(row.get("ground_clearance_m"), 1e6)
    near_count = safe_float(row.get("ground_near_point_count"), 0.0)
    shape_score = safe_float(row.get("ground_shape_score"), 0.0)
    near_ground_band_score = safe_float(row.get("near_ground_band_score"), 0.0)
    elevation_deg = abs(safe_float(row.get("geom_elevation_deg", row.get("elevation_deg", row.get("selected_pitch_deg"))), 0.0))
    range_m = safe_float(row.get("geom_range_m", row.get("range_m")), 1e6)
    score = 0.0
    score += 2.2 * min(overlap, 1.0)
    score += 0.7 if near_count >= 1.0 else 0.0
    score += 0.8 if clearance <= 20.0 else 0.0
    score += 0.5 if clearance <= 10.0 else 0.0
    score += 0.5 if elevation_deg <= 2.5 else 0.0
    score += 0.35 * min(shape_score, 3.0)
    score += 0.55 * min(near_ground_band_score, 4.5)
    score += 0.4 if range_m <= 500.0 and elevation_deg <= 5.0 else 0.0
    return sigmoid(score - 2.35)


def candidate_objectness(row: dict[str, str]) -> float:
    area = safe_float(row.get("geom_area_px", row.get("area_px")), 0.0)
    peak = safe_float(row.get("geom_peak_db", row.get("peak_db")), 0.0)
    mean = safe_float(row.get("geom_mean_db", row.get("mean_db")), 0.0)
    fill = safe_float(row.get("geom_fill_ratio", row.get("fill_ratio")), 0.0)
    active_layers = safe_float(row.get("radar_active_layer_count"), 0.0)
    grad = safe_float(row.get("radar_grad_mag_mean"), 0.0)
    gnd = groundness(row)
    score = 0.0
    score += min(area / 160.0, 1.8)
    score += max(0.0, (peak - 18.0) / 10.0)
    score += max(0.0, (mean - 12.0) / 12.0)
    score += 0.5 if 0.05 <= fill <= 0.95 else -0.3
    score += 0.4 if active_layers >= 1.0 else -0.4
    score += 0.3 if grad >= 2.0 else -0.2
    score -= 1.8 * gnd
    return sigmoid(score - 1.5)


def coarse_family(row: dict[str, str]) -> str:
    if safe_float(row.get("fam_powerline"), 0.0) > 0.5:
        return "line"
    if safe_float(row.get("fam_dense"), 0.0) > 0.5:
        return "dense"
    if safe_float(row.get("fam_isolated"), 0.0) > 0.5:
        return "isolated"
    az_span = safe_float(row.get("geom_az_span_deg"), 0.0)
    area = safe_float(row.get("geom_area_px"), 0.0)
    aspect = safe_float(row.get("radar_patch_aspect"), 0.0)
    if az_span >= 4.0 or aspect >= 2.5:
        return "line"
    if area >= 60.0:
        return "dense"
    return "isolated"


def rule_scores(row: dict[str, str]) -> dict[str, float]:
    family = coarse_family(row)
    g_az = safe_float(row.get("geom_az_span_deg"), 0.0)
    g_range = safe_float(row.get("geom_range_span_m"), 0.0)
    g_area = safe_float(row.get("geom_area_px"), 0.0)
    g_fill = safe_float(row.get("geom_fill_ratio"), 0.0)
    peak_db = safe_float(row.get("geom_peak_db"), 0.0)
    mean_db = safe_float(row.get("geom_mean_db"), 0.0)
    range_m = safe_float(row.get("geom_range_m"), 0.0)
    near_ground_flag = safe_float(row.get("near_ground_flag"), 0.0)

    aspect = safe_float(row.get("radar_patch_aspect"), 0.0)
    active_layers = safe_float(row.get("radar_active_layer_count"), 0.0)
    layer_ratio = safe_float(row.get("radar_selected_layer_energy_ratio"), 0.0)
    layer_width = safe_float(row.get("radar_layer_halfmax_width"), 0.0)
    layer_entropy = safe_float(row.get("radar_layer_energy_entropy"), 0.0)
    az_peak = safe_float(row.get("radar_az_profile_peak_ratio"), 0.0)
    rg_peak = safe_float(row.get("radar_range_profile_peak_ratio"), 0.0)
    az_entropy = safe_float(row.get("radar_az_profile_entropy"), 0.0)
    rg_entropy = safe_float(row.get("radar_range_profile_entropy"), 0.0)
    grad_r = safe_float(row.get("radar_grad_range_mean"), 0.0)
    grad_a = safe_float(row.get("radar_grad_az_mean"), 0.0)
    grad_mag = safe_float(row.get("radar_grad_mag_mean"), 0.0)
    hot_frac = safe_float(row.get("radar_hot_fraction"), 0.0)
    gnd = groundness(row)
    is_isolated = safe_float(row.get("fam_isolated"), 0.0) > 0.5
    is_far_isolated = is_isolated and range_m >= 1500.0
    is_high_elevation_far_isolated = is_far_isolated and safe_float(row.get("geom_elevation_deg"), 0.0) >= 4.0

    scores = {cls: 0.0 for cls in TARGET_CLASSES}

    scores["Power line"] += 2.6 if family == "line" else -0.6
    scores["Power line"] += 1.4 if aspect >= 3.0 else -0.4
    scores["Power line"] += 1.2 if g_range <= 35.0 else -0.5
    scores["Power line"] += 0.9 if active_layers <= 2.5 else -0.6
    scores["Power line"] += 0.8 if layer_ratio >= 0.35 else -0.2
    scores["Power line"] += 0.6 if az_entropy <= 0.75 else -0.2

    scores["Bridge"] += 2.0 if family == "line" else -0.4
    scores["Bridge"] += 1.4 if g_range >= 20.0 else -0.3
    scores["Bridge"] += 1.0 if g_area >= 18.0 else -0.2
    scores["Bridge"] += 0.9 if active_layers >= 2.0 else -0.2
    scores["Bridge"] += 0.7 if range_m >= 120.0 else 0.0
    scores["Bridge"] += 0.5 if hot_frac >= 0.18 else -0.1
    scores["Bridge"] -= 1.0 if near_ground_flag >= 0.5 and range_m <= 900.0 else 0.0

    scores["Building complex"] += 2.6 if family == "dense" else -0.5
    scores["Building complex"] += 1.4 if g_area >= 50.0 else -0.3
    scores["Building complex"] += 1.0 if active_layers >= 3.0 else -0.2
    scores["Building complex"] += 0.8 if layer_entropy >= 0.55 else -0.2
    scores["Building complex"] += 0.7 if g_az >= 2.5 else -0.1
    scores["Building complex"] += 0.6 if hot_frac >= 0.15 else -0.1

    scores["Tall building"] += 2.0 if family in {"dense", "isolated"} else -0.3
    scores["Tall building"] += 1.3 if active_layers >= 3.0 else -0.2
    scores["Tall building"] += 1.0 if layer_width >= 2.0 else -0.2
    scores["Tall building"] += 0.8 if g_az <= 3.5 else -0.2
    scores["Tall building"] += 0.6 if rg_peak >= 0.10 else -0.1
    scores["Tall building"] += 0.5 if grad_r >= grad_a else -0.1
    scores["Tall building"] -= 1.0 if is_far_isolated else 0.0
    scores["Tall building"] -= 0.9 if is_high_elevation_far_isolated else 0.0
    scores["Tall building"] -= 0.7 if is_high_elevation_far_isolated and g_az >= 1.5 else 0.0
    scores["Tall building"] -= 0.6 if is_far_isolated and layer_entropy >= 0.60 else 0.0

    scores["Chimney"] += 2.1 if family == "isolated" else -0.4
    scores["Chimney"] += 1.2 if g_area <= 18.0 else -0.2
    scores["Chimney"] += 1.0 if active_layers <= 2.0 else -0.4
    scores["Chimney"] += 0.8 if hot_frac >= 0.18 else -0.1
    scores["Chimney"] += 0.7 if az_peak >= 0.25 else -0.2
    scores["Chimney"] += 0.5 if layer_ratio >= 0.45 else -0.1

    scores["Power tower"] += 2.2 if family == "isolated" else -0.4
    scores["Power tower"] += 1.2 if active_layers >= 2.0 else -0.2
    scores["Power tower"] += 1.0 if grad_a >= 2.0 else -0.2
    scores["Power tower"] += 0.9 if grad_r >= 2.0 else -0.2
    scores["Power tower"] += 0.8 if layer_width >= 2.0 else -0.2
    scores["Power tower"] += 0.6 if 8.0 <= g_area <= 45.0 else -0.1

    scores["Signal tower"] += 2.0 if family == "isolated" else -0.3
    scores["Signal tower"] += 1.1 if g_area <= 24.0 else -0.2
    scores["Signal tower"] += 1.0 if active_layers <= 3.0 else -0.2
    scores["Signal tower"] += 0.9 if layer_ratio >= 0.42 else -0.2
    scores["Signal tower"] += 0.7 if az_peak >= 0.22 else -0.1
    scores["Signal tower"] += 0.5 if layer_entropy <= 0.70 else -0.1

    scores["Wind turbine"] += 2.0 if family == "isolated" else -0.3
    scores["Wind turbine"] += 1.2 if active_layers >= 3.0 else -0.3
    scores["Wind turbine"] += 1.0 if layer_entropy >= 0.65 else -0.2
    scores["Wind turbine"] += 0.9 if 0.10 <= hot_frac <= 0.28 else -0.2
    scores["Wind turbine"] += 0.7 if abs(grad_r - grad_a) <= 0.8 else -0.2
    scores["Wind turbine"] += 0.6 if az_entropy >= 0.75 else -0.2
    scores["Wind turbine"] += 1.1 if is_far_isolated else 0.0
    scores["Wind turbine"] += 1.0 if is_high_elevation_far_isolated else 0.0
    scores["Wind turbine"] += 0.9 if is_high_elevation_far_isolated and 1.0 <= g_az <= 6.0 else 0.0
    scores["Wind turbine"] += 0.7 if is_far_isolated and layer_entropy >= 0.60 else 0.0
    scores["Wind turbine"] += 0.6 if is_far_isolated and abs(grad_r - grad_a) <= 1.2 else 0.0
    scores["Wind turbine"] += 0.5 if is_far_isolated and active_layers >= 2.0 else 0.0

    if row.get("weak_label") == "Power line":
        scores["Power line"] += 2.0
        scores["Bridge"] += 0.4
    if row.get("weak_label") == "Building complex":
        scores["Building complex"] += 2.0
        scores["Tall building"] += 0.6

    for cls in TARGET_CLASSES:
        scores[cls] += 0.14 * sigmoid((peak_db - 8.0) / 4.0)
        scores[cls] += 0.10 * sigmoid((mean_db - 5.0) / 3.0)
        scores[cls] -= 1.25 * gnd

    if gnd >= 0.70:
        scores["Building complex"] -= 0.8
        scores["Tall building"] -= 0.8
        scores["Chimney"] -= 0.6
        scores["Power tower"] -= 0.6
        scores["Signal tower"] -= 0.6
        scores["Wind turbine"] -= 0.6
        scores["Bridge"] -= 0.4

    return scores


def estimate_sensor_xyz(row: dict[str, str]) -> tuple[float, float, float]:
    range_m = safe_float(row.get("geom_range_m", row.get("range_m")), 0.0)
    az_deg = safe_float(row.get("geom_azimuth_deg", row.get("azimuth_deg")), 0.0)
    pitch_deg = safe_float(row.get("geom_elevation_deg", row.get("selected_pitch_deg")), 0.0)
    az = math.radians(az_deg)
    el = math.radians(pitch_deg)
    x_forward = range_m * math.cos(el) * math.cos(az)
    y_left = range_m * math.cos(el) * math.sin(az)
    z_up = range_m * math.sin(el)
    return x_forward, y_left, z_up


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("no prediction rows to write")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    feature_csv = args.feature_csv.resolve()
    out_csv = args.out_csv.resolve() if args.out_csv else feature_csv.parent / "predictions.csv"

    rows = load_csv_rows(feature_csv)
    if not rows:
        raise RuntimeError(f"no feature rows found in {feature_csv}")

    out_rows: list[dict[str, str]] = []
    for row in rows:
        scores = rule_scores(row)
        probs = softmax(scores)
        obj = candidate_objectness(row)
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        conf = float(probs[label] * obj)
        x_forward, y_left, z_up = estimate_sensor_xyz(row)

        out = dict(row)
        out["pred_label"] = label
        out["pred_confidence"] = f"{conf:.6f}"
        out["pred_objectness"] = f"{obj:.6f}"
        out["pred_groundness"] = f"{groundness(row):.6f}"
        out["x_forward_m"] = f"{x_forward:.6f}"
        out["y_left_m"] = f"{y_left:.6f}"
        out["z_up_m"] = f"{z_up:.6f}"
        for cls in TARGET_CLASSES:
            out[f"score_{cls}"] = f"{scores[cls]:.6f}"
            out[f"prob_{cls}"] = f"{probs[cls]:.6f}"
        out_rows.append(out)

    write_predictions(out_csv, out_rows)
    print(f"feature csv: {feature_csv}")
    print(f"prediction csv: {out_csv}")
    print(f"prediction rows: {len(out_rows)}")
    print("[NOTE] This is a radar-only rule-based heuristic recognizer.")


if __name__ == "__main__":
    main()
