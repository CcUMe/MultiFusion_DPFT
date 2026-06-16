from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


TARGET_CLASSES = [
    "Bridge",
    "Building complex",
    "Chimney",
    "Power line",
    "Power tower",
    "Signal tower",
    "Tall building",
    "Wind turbine",
]

NON_FEATURE_COLUMNS = {
    "candidate_id",
    "image",
    "image_path",
    "ra",
    "ra_path",
    "part_name",
    "candidate_family",
    "weak_label",
    "label",
    "split",
}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def power_to_db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float32), 1e-10))


def parse_pitch_layers(text: str) -> list[float]:
    if not text:
        return []
    vals = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except Exception:
            pass
    return vals


def choose_layer_index(row: dict[str, str]) -> int:
    idx = safe_int(row.get("selected_pitch_idx"), -1)
    count = safe_int(row.get("pitch_layer_count"), 0)
    if 0 <= idx < count:
        return idx
    if count > 0:
        return count // 2
    return 0


def numeric_feature_columns(rows: list[dict[str, str]], label_column: str) -> list[str]:
    if not rows:
        return []
    cols = [c for c in rows[0].keys() if c not in NON_FEATURE_COLUMNS and c != label_column]
    numeric_cols: list[str] = []
    for c in cols:
        ok = True
        for row in rows[: min(20, len(rows))]:
            val = row.get(c, "")
            if val in {"", None}:
                continue
            try:
                float(val)
            except Exception:
                ok = False
                break
        if ok:
            numeric_cols.append(c)
    return numeric_cols




def heuristic_image_roi(
    image_w: int,
    image_h: int,
    azimuth_deg: float,
    range_m: float,
    az_min_deg: float,
    az_max_deg: float,
    range_max_m: float,
    box_scale_x: float = 0.18,
    box_scale_y: float = 0.18,
) -> tuple[int, int, int, int]:
    if az_max_deg <= az_min_deg:
        az_max_deg = az_min_deg + 1.0
    if range_max_m <= 0:
        range_max_m = 1.0

    x_frac = (azimuth_deg - az_min_deg) / (az_max_deg - az_min_deg)
    x_frac = min(max(x_frac, 0.0), 1.0)

    range_frac = min(max(range_m / range_max_m, 0.0), 1.0)
    y_frac = 0.15 + 0.75 * (1.0 - range_frac)
    y_frac = min(max(y_frac, 0.05), 0.95)

    cx = x_frac * (image_w - 1)
    cy = y_frac * (image_h - 1)
    half_w = max(24.0, image_w * box_scale_x * (0.6 + 0.4 * (1.0 - range_frac)))
    half_h = max(24.0, image_h * box_scale_y * (0.6 + 0.4 * (1.0 - range_frac)))

    x0 = int(max(0, round(cx - half_w)))
    x1 = int(min(image_w, round(cx + half_w)))
    y0 = int(max(0, round(cy - half_h)))
    y1 = int(min(image_h, round(cy + half_h)))
    if x1 <= x0:
        x1 = min(image_w, x0 + 1)
    if y1 <= y0:
        y1 = min(image_h, y0 + 1)
    return x0, y0, x1, y1


def compute_range_azimuth(
    range_idx: float,
    az_idx: float,
    n_range: int,
    n_az: int,
    range_max_m: float,
    az_min_deg: float,
    az_max_deg: float,
) -> tuple[float, float]:
    if n_range <= 1:
        range_m = 0.0
    else:
        range_m = float(range_idx) / float(n_range - 1) * range_max_m
    if n_az <= 1:
        az_deg = 0.0
    else:
        az_deg = az_min_deg + float(az_idx) / float(n_az - 1) * (az_max_deg - az_min_deg)
    return range_m, az_deg


def parse_projection_k(calib_path: Path) -> np.ndarray:
    lines = [line.strip() for line in calib_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    idx = next(i for i, line in enumerate(lines) if line.lower() == "projection")
    row0 = [float(x) for x in lines[idx + 1].split()]
    row1 = [float(x) for x in lines[idx + 2].split()]
    return np.array(
        [[row0[0], 0.0, row0[2]], [0.0, row1[1], row1[2]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def infer_camera_id(image_name: str, image_path: Path | None = None) -> str | None:
    candidates = [image_name]
    if image_path is not None:
        candidates.extend([image_path.name, str(image_path)])
    for text in candidates:
        match = re.search(r"(DA\d{7,})", text)
        if match:
            return match.group(1)
    return None


def infer_camera_side(image_name: str, image_path: Path | None = None) -> str | None:
    camera_id = infer_camera_id(image_name, image_path)
    if camera_id == "DA8679038":
        return "right"
    if camera_id == "DA8679037":
        return "left"
    text = f"{image_name} {image_path or ''}".lower()
    if "right" in text or "右" in text:
        return "right"
    if "left" in text or "左" in text:
        return "left"
    return None


def resolve_shared_calib_path(
    cap_dir: Path,
    part_name: str = "",
    image_name: str = "",
    image_path: Path | None = None,
    explicit_calib_path: Path | None = None,
) -> Path | None:
    if explicit_calib_path is not None and explicit_calib_path.exists():
        return explicit_calib_path

    camera_id = infer_camera_id(image_name, image_path)
    camera_side = infer_camera_side(image_name, image_path)
    preferred_names: list[str] = []
    if camera_id:
        preferred_names.extend([
            f"{camera_id}.txt",
            f"{camera_id}_tele.txt",
            f"hikrobot_{camera_id}_tele.txt",
        ])
    if camera_side:
        preferred_names.append(f"{camera_side}cam.txt")
    preferred_names.extend(["rightcam.txt", "leftcam.txt"])

    search_roots: list[Path] = []
    if image_path is not None:
        search_roots.extend(image_path.parents)
    if part_name:
        search_roots.append(cap_dir / part_name)
    search_roots.append(cap_dir)
    search_roots.extend(cap_dir.parents)

    seen_roots: set[str] = set()
    for root in search_roots:
        root_key = str(root)
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        for name in preferred_names:
            direct = root / name
            if direct.exists():
                return direct
        if root.exists() and root.is_dir():
            for name in preferred_names:
                matches = sorted(root.glob(name))
                if matches:
                    return matches[0]
    return None


def r_radar_to_body_yaw(yaw_deg: float) -> np.ndarray:
    y = np.radians(yaw_deg)
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def r_cam_from_tilt(tilt_deg: float) -> np.ndarray:
    t = np.radians(tilt_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[0.0, -1.0, 0.0], [-s, 0.0, -c], [c, 0.0, -s]], dtype=np.float64)


def radar_points_body(ranges: np.ndarray, az_deg: np.ndarray, el_deg: float) -> np.ndarray:
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    ce, se = np.cos(el), np.sin(el)
    x_fwd = ranges * ce * np.cos(az)
    y_left = -ranges * ce * np.sin(az)
    z_up = ranges * se
    return np.stack([x_fwd, y_left, z_up], axis=0)


def project_radar_points_to_image(
    points_radar: np.ndarray,
    k: np.ndarray,
    image_size: tuple[int, int],
    tilt_deg: float = 0.0,
    yaw_deg: float = 0.0,
    lever_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_yaw = r_radar_to_body_yaw(yaw_deg)
    p_body = r_yaw @ points_radar + np.asarray(lever_xyz, dtype=np.float64).reshape(3, 1)
    p_opt = r_cam_from_tilt(tilt_deg) @ p_body
    z = p_opt[2]
    valid = z > 0.1
    u = np.full(z.shape, np.nan, dtype=np.float64)
    v = np.full(z.shape, np.nan, dtype=np.float64)
    u[valid] = k[0, 0] * p_opt[0, valid] / z[valid] + k[0, 2]
    v[valid] = k[1, 1] * p_opt[1, valid] / z[valid] + k[1, 2]
    w, h = image_size
    mask = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, mask
