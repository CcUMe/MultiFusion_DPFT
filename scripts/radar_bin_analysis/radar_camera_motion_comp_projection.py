#!/usr/bin/env python
"""Bag 1.1 mmWave radar -> camera projection with bin-pose motion compensation.

This is the deliverable, standalone version of the radar projection workflow.
It uses only the Bag 1.1 files:

  data/bag1.1/with_cameras_capture_20260427_151113_mmwave_udp.bin
  data/bag1.1/segment_000_000076.000_000102.000/image_to_antframe_time_aligned.csv
  data/bag1.1/segment_000_000076.000_000102.000/nav100_state/nav100__state/nav100__state.csv
  data/bag1.1/segment_000_000076.000_000102.000/rightcam.txt
  data/mmwave_mat_1218style/*.mat

For one source camera frame, it:
  1. finds the matched AntFrame .mat;
  2. reads per-beam GPS/heading from the raw UDP .bin using mat-local FZ/raw index;
  3. detects radar returns with CA-CFAR along range;
  4. builds motion-compensated ENU radar points;
  5. projects those points into nearby camera frames;
  6. writes overlay images, sparse depth CSVs, and a contact sheet.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
BAG_DIR = ROOT / "data" / "bag1.1"
SEG_DIR = BAG_DIR / "segment_000_000076.000_000102.000"
BIN_PATH = BAG_DIR / "with_cameras_capture_20260427_151113_mmwave_udp.bin"
MATCH_CSV = SEG_DIR / "image_to_antframe_time_aligned.csv"
NAV_CSV = SEG_DIR / "nav100_state" / "nav100__state" / "nav100__state.csv"
CALIB_TXT = SEG_DIR / "rightcam.txt"
MAT_DIR = ROOT / "data" / "mmwave_mat_1218style"
DEFAULT_IMAGE_DIR = SEG_DIR / "images" / "hikrobot_camera__DA8679037__image_raw_stablelized_1920x1200"
OUT_DIR = ROOT / "outputs" / "radar_camera_motion_comp_projection"

FRAME_BYTES = 8624
RADAR_RANGE_STEP_M = 6.0
CFAR_TRAIN = 15
CFAR_GUARD = 2
CFAR_PFA = 1e-4
SKY_V_MIN_PX = 400
R_EARTH_EQ = 6378137.0
R_EARTH_POL = 6356752.314

# Camera -> radar lever arm in body FLU: x forward, y left, z up.
# Default is zero for delivery. Pass calibrated values with --lever-x/y/z.
LEVER_CAM2RADAR = np.array([0.0, 0.0, 0.0], dtype=np.float64)


def natural_key(text: str) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def cfar_alpha(train_each_side: int, pfa: float) -> float:
    n = train_each_side * 2
    return n * (pfa ** (-1.0 / n) - 1.0)


CFAR_ALPHA = cfar_alpha(CFAR_TRAIN, CFAR_PFA)


def ca_cfar_1d(power_lin: np.ndarray) -> np.ndarray:
    n = power_lin.shape[0]
    half_win = CFAR_TRAIN + CFAR_GUARD
    cs = np.concatenate(([0.0], np.cumsum(power_lin, dtype=np.float64)))
    idx = np.arange(n)
    l0 = np.maximum(0, idx - half_win)
    l1 = np.maximum(0, idx - CFAR_GUARD)
    r0 = np.minimum(n, idx + CFAR_GUARD + 1)
    r1 = np.minimum(n, idx + half_win + 1)
    n_train = (l1 - l0) + (r1 - r0)
    noise_sum = (cs[l1] - cs[l0]) + (cs[r1] - cs[r0])
    noise = noise_sum / np.maximum(n_train, 1)
    noise = np.where(n_train > 0, noise, power_lin)
    return power_lin > (CFAR_ALPHA * noise)


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
        key = str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
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


def load_nav_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def interp_nav(nav: np.ndarray, t_bag: float) -> dict[str, float]:
    ts = nav["relative_time_sec"].astype(np.float64)
    i = int(np.clip(np.searchsorted(ts, t_bag), 1, len(ts) - 1))
    denom = max(float(ts[i] - ts[i - 1]), 1e-9)
    w = float(np.clip((t_bag - ts[i - 1]) / denom, 0.0, 1.0))

    def lerp(field: str) -> float:
        return float(nav[field][i - 1]) + w * (float(nav[field][i]) - float(nav[field][i - 1]))

    return {
        "lat": lerp("latitude"),
        "lng": lerp("longitude"),
        "alt": lerp("altitude"),
        "pitch": lerp("pitch"),
        "roll": lerp("roll"),
        "heading": lerp("true_heading_deg"),
    }


def load_match_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            if row.get("assignment") not in {"inside", "nearest", ""}:
                continue
            rows.append(row)
    rows.sort(key=lambda r: int(r["camera_seq"]))
    return rows


def row_by_seq(rows: list[dict], seq: int) -> dict:
    for row in rows:
        if int(row["camera_seq"]) == seq:
            return row
    valid = ", ".join(str(int(r["camera_seq"])) for r in rows[:10])
    raise ValueError(f"camera seq {seq} not found in {MATCH_CSV}. First valid seqs: {valid} ...")


def select_target_rows(rows: list[dict], source_t_bag: float, time_window_s: float, max_frames: int) -> list[dict]:
    selected = [r for r in rows if abs(float(r["camera_t_bag"]) - source_t_bag) <= time_window_s + 1e-9]
    selected.sort(key=lambda r: (abs(float(r["camera_t_bag"]) - source_t_bag), int(r["camera_seq"])))
    if max_frames > 0:
        selected = selected[:max_frames]
    return sorted(selected, key=lambda r: int(r["camera_seq"]))


def image_path_for_row(row: dict, image_dir: Path, match_csv: Path) -> Path:
    name = row["image"]
    exact = image_dir / name
    if exact.exists():
        return exact
    seq = int(row["camera_seq"])
    candidates = sorted(image_dir.glob(f"*_{seq:06d}_*.jpg"), key=lambda p: natural_key(p.name))
    if candidates:
        return candidates[0]
    raw = match_csv.parent / "images" / "hikrobot_camera__DA8679037__image_raw" / name
    if raw.exists():
        return raw
    raise FileNotFoundError(f"cannot find image for seq={seq}, name={name}, image_dir={image_dir}")


def load_raw_pose_by_raw_idx(bin_path: Path, raw_idx_s: int, raw_idx_e: int) -> dict[int, dict]:
    poses: dict[int, dict] = {}
    n_raw = bin_path.stat().st_size // FRAME_BYTES
    raw_idx_s = max(0, int(raw_idx_s))
    raw_idx_e = min(n_raw - 1, int(raw_idx_e))
    with bin_path.open("rb") as f:
        for idx in range(raw_idx_s, raw_idx_e + 1):
            f.seek(idx * FRAME_BYTES)
            raw = f.read(FRAME_BYTES)
            if len(raw) < FRAME_BYTES:
                break
            if struct.unpack_from("<I", raw, 0)[0] != 0xABABABAB:
                continue
            net_fz = struct.unpack_from("<I", raw, 12)[0]
            lon, lat = struct.unpack_from("<ff", raw, 48)
            heading = struct.unpack_from("<f", raw, 56)[0]
            poses[idx] = {
                "lat": float(lat),
                "lon": float(lon),
                "heading": float(heading),
                "raw_idx": int(idx),
                "net_fz": int(net_fz),
            }
    return poses


def nearest_pose(poses: dict[int, dict], key: int) -> dict | None:
    if not poses:
        return None
    if key in poses:
        return poses[key]
    return poses[min(poses.keys(), key=lambda k: abs(k - key))]


def gps_to_enu_m(lat_deg: float, lon_deg: float, ref_lat_deg: float, ref_lon_deg: float) -> np.ndarray:
    coslat = np.cos(np.radians(ref_lat_deg))
    east = (lon_deg - ref_lon_deg) * np.radians(1.0) * R_EARTH_EQ * coslat
    north = (lat_deg - ref_lat_deg) * np.radians(1.0) * R_EARTH_POL
    return np.array([east, north, 0.0], dtype=np.float64)


def body_to_enu(heading_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    psi = np.radians(heading_deg)
    th = np.radians(pitch_deg)
    ph = np.radians(roll_deg)
    sp, cp = np.sin(psi), np.cos(psi)
    st, ct = np.sin(th), np.cos(th)
    sr, cr = np.sin(ph), np.cos(ph)
    r_yaw = np.array([[sp, -cp, 0.0], [cp, sp, 0.0], [0.0, 0.0, 1.0]])
    r_pitch = np.array([[ct, 0.0, -st], [0.0, 1.0, 0.0], [st, 0.0, ct]])
    r_roll = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return r_yaw @ r_pitch @ r_roll


def r_body_to_enu_heading(heading_deg: float) -> np.ndarray:
    h = np.radians(heading_deg)
    s, c = np.sin(h), np.cos(h)
    return np.array([[s, -c, 0.0], [c, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


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


def angle_delta_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def build_source_world_points(
    bin_path: Path,
    mat_path: Path,
    fz_start: int,
    fz_end: int,
    source_t_bag: float,
    nav: np.ndarray,
    yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    mat = sio.loadmat(str(mat_path))
    data = mat["Data_Ori"]
    raw_poses = load_raw_pose_by_raw_idx(bin_path, fz_start, fz_end)
    source_nav = interp_nav(nav, source_t_bag)
    center_idx = (fz_start + fz_end) // 2
    center_pose = nearest_pose(raw_poses, center_idx) or {
        "lat": float(source_nav["lat"]),
        "lon": float(source_nav["lng"]),
        "heading": float(source_nav["heading"]),
    }
    r_yaw = r_radar_to_body_yaw(yaw_deg)
    ranges_axis = np.arange(1, data[0][0][0][2].shape[0] + 1, dtype=np.float64) * RADAR_RANGE_STEP_M

    pts_enu, ranges, heights_abs, powers, beam_fzs = [], [], [], [], []
    beam_idx = 0
    for ei in range(data.shape[0]):
        cell = data[ei][0][0]
        el_deg = float(cell[0][0, 0])
        az_arr = cell[1][0]
        pwr_db = np.maximum(cell[2].astype(np.float64), cell[3].astype(np.float64))
        for j, az_raw in enumerate(az_arr):
            beam_fz = fz_start + beam_idx
            beam_idx += 1
            beam_pose = nearest_pose(raw_poses, beam_fz)
            col_db = pwr_db[:, j]
            cfar_ok = ca_cfar_1d(np.power(10.0, col_db / 10.0))
            valid = cfar_ok
            if not valid.any():
                continue
            vi = np.where(valid)[0]
            vi = np.array([vi[int(np.argmax(col_db[vi]))]], dtype=np.int64)
            r_vals = ranges_axis[vi]
            p_vals = col_db[vi]
            p_radar = radar_points_body(r_vals, np.full(len(r_vals), float(az_raw)), el_deg)

            if beam_pose is not None:
                beam_offset = gps_to_enu_m(
                    float(beam_pose["lat"]),
                    float(beam_pose["lon"]),
                    float(center_pose["lat"]),
                    float(center_pose["lon"]),
                )
                r_beam = r_body_to_enu_heading(float(beam_pose["heading"]))
                p_world = beam_offset[:, None] + r_beam @ (r_yaw @ p_radar)
                p_body_source = r_body_to_enu_heading(float(center_pose["heading"])).T @ p_world + LEVER_CAM2RADAR[:, None]
            else:
                p_world = r_body_to_enu_heading(float(center_pose["heading"])) @ (r_yaw @ p_radar)
                p_body_source = r_yaw @ p_radar + LEVER_CAM2RADAR[:, None]

            h_abs = p_body_source[2] + float(source_nav["alt"])
            pts_enu.append(p_world.T)
            ranges.append(r_vals)
            heights_abs.append(h_abs)
            powers.append(p_vals)
            beam_fzs.append(np.full(len(r_vals), beam_fz, dtype=np.int64))

    if not pts_enu:
        empty = np.array([])
        return np.empty((0, 3)), empty, empty, empty, empty, center_pose
    return (
        np.vstack(pts_enu),
        np.concatenate(ranges),
        np.concatenate(heights_abs),
        np.concatenate(powers),
        np.concatenate(beam_fzs),
        center_pose,
    )


def build_source_world_points_from_rae(
    ra_path: Path,
    source_row: dict,
    source_t_bag: float,
    nav: np.ndarray,
    yaw_deg: float,
    threshold_percentile: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    rae = np.load(ra_path, allow_pickle=False)
    if rae.ndim != 3 or rae.shape[2] == 0:
        raise RuntimeError(f"invalid rae tensor: {ra_path} shape={getattr(rae, 'shape', None)}")

    layer_idx = 0
    try:
        layer_idx = int(float(source_row.get("selected_pitch_idx", 0) or 0))
    except Exception:
        layer_idx = 0
    layer_idx = min(max(layer_idx, 0), rae.shape[2] - 1)
    pitch_deg = float(source_row.get("selected_pitch_deg", 0.0) or 0.0)
    range_max_m = float(source_row.get("range_max_m", rae.shape[0]) or rae.shape[0])
    az_min_deg = float(source_row.get("az_min_deg", -60.0) or -60.0)
    az_max_deg = float(source_row.get("az_max_deg", 60.0) or 60.0)

    ra_slice = np.asarray(rae[:, :, layer_idx], dtype=np.float32)
    ra_db = 20.0 * np.log10(np.maximum(ra_slice, 1e-10))
    finite = np.isfinite(ra_db)
    if not finite.any():
        raise RuntimeError(f"no finite ra values in {ra_path}")
    thr = float(np.percentile(ra_db[finite], threshold_percentile))
    mask = finite & (ra_db >= thr)
    coords = np.argwhere(mask)
    if coords.size == 0:
        topk = np.argpartition(ra_db.ravel(), -min(max_points, ra_db.size))[-min(max_points, ra_db.size):]
        coords = np.column_stack(np.unravel_index(topk, ra_db.shape))
    if coords.shape[0] > max_points > 0:
        vals = ra_db[coords[:, 0], coords[:, 1]]
        keep = np.argpartition(vals, -max_points)[-max_points:]
        coords = coords[keep]

    n_range, n_az = ra_slice.shape
    ranges = np.zeros(coords.shape[0], dtype=np.float64)
    azs = np.zeros(coords.shape[0], dtype=np.float64)
    for i, (r_idx, a_idx) in enumerate(coords):
        ranges[i] = 0.0 if n_range <= 1 else float(r_idx) / float(n_range - 1) * range_max_m
        azs[i] = 0.0 if n_az <= 1 else az_min_deg + float(a_idx) / float(n_az - 1) * (az_max_deg - az_min_deg)

    powers = ra_db[coords[:, 0], coords[:, 1]].astype(np.float64)
    source_nav = interp_nav(nav, source_t_bag)
    center_pose = {
        "lat": float(source_nav["lat"]),
        "lon": float(source_nav["lng"]),
        "heading": float(source_nav["heading"]),
    }
    p_radar = radar_points_body(ranges, azs, pitch_deg)
    p_world = (r_body_to_enu_heading(float(center_pose["heading"])) @ (r_radar_to_body_yaw(yaw_deg) @ p_radar)).T
    p_body_source = r_radar_to_body_yaw(yaw_deg) @ p_radar + LEVER_CAM2RADAR[:, None]
    heights_abs = p_body_source[2] + float(source_nav["alt"])
    beam_fzs = np.full(coords.shape[0], int(float(source_row.get("pkt_start", 0) or 0)), dtype=np.int64)
    return p_world, ranges, heights_abs, powers, beam_fzs, center_pose


def project_world_to_image(
    p_world: np.ndarray,
    center_pose: dict,
    source_nav: dict,
    target_nav: dict,
    k: np.ndarray,
    image_size: tuple[int, int],
    tilt_deg: float,
    use_pitch_roll: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_offset = gps_to_enu_m(
        float(target_nav["lat"]),
        float(target_nav["lng"]),
        float(source_nav["lat"]),
        float(source_nav["lng"]),
    )
    target_offset[2] = float(target_nav["alt"]) - float(source_nav["alt"])
    heading_eff = float(center_pose["heading"]) + angle_delta_deg(
        float(target_nav["heading"]), float(source_nav["heading"])
    )
    if use_pitch_roll:
        r_target = body_to_enu(
            heading_eff,
            math.degrees(float(target_nav["pitch"]) - float(source_nav["pitch"])),
            math.degrees(float(target_nav["roll"]) - float(source_nav["roll"])),
        )
    else:
        r_target = r_body_to_enu_heading(heading_eff)
    p_body = r_target.T @ (p_world.T - target_offset[:, None]) + LEVER_CAM2RADAR[:, None]
    p_opt = r_cam_from_tilt(tilt_deg) @ p_body
    z = p_opt[2]
    valid = z > 0.1
    u = np.full(z.shape, np.nan)
    v = np.full(z.shape, np.nan)
    u[valid] = k[0, 0] * p_opt[0, valid] / z[valid] + k[0, 2]
    v[valid] = k[1, 1] * p_opt[1, valid] / z[valid] + k[1, 2]
    w, h = image_size
    mask = valid & (u >= 0) & (u < w) & (v >= SKY_V_MIN_PX) & (v < h)
    return u, v, mask, z


def write_depth_csv(
    path: Path,
    u: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    ranges: np.ndarray,
    heights_abs: np.ndarray,
    powers: np.ndarray,
    beam_fzs: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["u", "v", "depth_m", "range_m", "height_abs_m", "power_db", "beam_raw_fz"])
        for vals in zip(u[mask], v[mask], depth[mask], ranges[mask], heights_abs[mask], powers[mask], beam_fzs[mask]):
            writer.writerow([
                f"{float(vals[0]):.3f}",
                f"{float(vals[1]):.3f}",
                f"{float(vals[2]):.3f}",
                f"{float(vals[3]):.3f}",
                f"{float(vals[4]):.3f}",
                f"{float(vals[5]):.2f}",
                int(vals[6]),
            ])


def visible_roi_from_projection(
    u: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    image_size: tuple[int, int],
    pad_px: int = 24,
) -> tuple[int, int, int, int, int, float] | None:
    if not mask.any():
        return None
    uu = u[mask]
    vv = v[mask]
    w, h = image_size
    x0 = int(max(0, np.floor(np.min(uu) - pad_px)))
    y0 = int(max(0, np.floor(np.min(vv) - pad_px)))
    x1 = int(min(w, np.ceil(np.max(uu) + pad_px)))
    y1 = int(min(h, np.ceil(np.max(vv) + pad_px)))
    if x1 <= x0 or y1 <= y0:
        return None
    visible_points = int(mask.sum())
    coverage = float(((x1 - x0) * (y1 - y0)) / max(1, w * h))
    return x0, y0, x1, y1, visible_points, coverage


def write_roi_constraints_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_seq", "camera_seq", "image", "image_path", "camera_t_bag", "dt_sec",
        "calib_path", "tilt_deg", "yaw_deg", "lever_x_m", "lever_y_m", "lever_z_m",
        "roi_x0", "roi_y0", "roi_x1", "roi_y1", "visible_points", "coverage_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_overlay(
    image_path: Path,
    u: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    ranges: np.ndarray,
    heights_abs: np.ndarray,
    powers: np.ndarray,
    title: str,
    out_path: Path,
    label_limit: int,
) -> None:
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(12, 7.5), dpi=140)
    ax.imshow(img)
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.axis("off")
    if mask.any():
        sc = ax.scatter(
            u[mask],
            v[mask],
            c=depth[mask],
            cmap="plasma_r",
            s=16,
            alpha=0.88,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.01)
        cbar.set_label("camera depth Z (m)")
        visible_idx = np.where(mask)[0]
        ordered = visible_idx[np.argsort(ranges[mask])]
        limit = len(ordered) if label_limit <= 0 else min(label_limit, len(ordered))
        used_cells: set[tuple[int, int]] = set()
        for idx in ordered[:limit]:
            cell = (int(u[idx]) // 36, int(v[idx]) // 36)
            if cell in used_cells:
                continue
            used_cells.add(cell)
            ax.text(
                u[idx] + 4,
                v[idx] - 4,
                f"D{depth[idx]:.0f}m\nH{heights_abs[idx]:.0f}m",
                fontsize=5.5,
                color="white",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.4, foreground="black")],
                clip_on=True,
            )
    ax.set_title(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_contact_sheet(image_paths: list[Path], titles: list[str], out_path: Path) -> None:
    if not image_paths:
        return
    cols = min(3, len(image_paths))
    rows = int(math.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.6, rows * 3.7), dpi=140)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr:
        ax.axis("off")
    for ax, img_path, title in zip(axes_arr, image_paths, titles):
        ax.imshow(Image.open(img_path).convert("RGB"))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone radar-to-camera projection with bin motion compensation")
    parser.add_argument("--source-seq", type=int, default=711, help="camera sequence used to select the source AntFrame")
    parser.add_argument("--time-window-s", type=float, default=0.5, help="nearby camera frames to render around source time")
    parser.add_argument("--max-frames", type=int, default=9)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--match-csv", type=Path, default=MATCH_CSV)
    parser.add_argument("--nav-csv", type=Path, default=NAV_CSV)
    parser.add_argument("--bin-path", type=Path, default=BIN_PATH)
    parser.add_argument("--mat-dir", type=Path, default=MAT_DIR)
    parser.add_argument("--ra-dir", type=Path, help="RAE npy directory for newer aligned CSVs without mat columns")
    parser.add_argument("--ra-threshold-percentile", type=float, default=99.7, help="Percentile threshold when building source points from ra.npy fallback")
    parser.add_argument("--max-ra-points", type=int, default=2500, help="Maximum number of fallback ra.npy source points after thresholding")
    parser.add_argument("--cap-dir", type=Path, help="capture root used for shared calibration lookup")
    parser.add_argument("--calib-path", type=Path, help="optional explicit calibration txt path")
    parser.add_argument("--tilt", type=float, default=0.0, help="camera downward tilt in degrees")
    parser.add_argument("--yaw", type=float, default=0.0, help="radar yaw offset in body frame, positive points left")
    parser.add_argument("--lever-x", type=float, default=0.0, help="camera-to-radar lever arm x/forward in meters")
    parser.add_argument("--lever-y", type=float, default=0.0, help="camera-to-radar lever arm y/left in meters")
    parser.add_argument("--lever-z", type=float, default=0.0, help="camera-to-radar lever arm z/up in meters")
    parser.add_argument("--label-limit", type=int, default=80)
    parser.add_argument("--roi-pad", type=int, default=24, help="padding in pixels when converting projected radar points into an image ROI constraint")
    parser.add_argument("--roi-csv", type=Path, help="optional output CSV path for projected visible-region ROI constraints")
    parser.add_argument("--camera-attitude", action="store_true", help="also use pitch/roll delta between source and target camera frames")
    args = parser.parse_args()

    LEVER_CAM2RADAR[:] = [args.lever_x, args.lever_y, args.lever_z]

    match_csv = args.match_csv.resolve()
    nav_csv = args.nav_csv.resolve()
    bin_path = args.bin_path.resolve()
    mat_dir = args.mat_dir.resolve()
    ra_dir = args.ra_dir.resolve() if args.ra_dir else ((args.cap_dir.resolve() if args.cap_dir else match_csv.parent) / "mmwave_ra_npy")
    image_dir = args.image_dir.resolve()
    out_dir = args.out_dir.resolve()
    cap_dir = args.cap_dir.resolve() if args.cap_dir else match_csv.parent

    nav = load_nav_csv(nav_csv)
    rows = load_match_rows(match_csv)
    source_row = row_by_seq(rows, args.source_seq)
    target_rows = select_target_rows(rows, float(source_row["camera_t_bag"]), args.time_window_s, args.max_frames)
    if not target_rows:
        raise RuntimeError("no target frames selected")

    source_image_path = image_path_for_row(source_row, image_dir, match_csv)
    calib_path = resolve_shared_calib_path(
        cap_dir=cap_dir,
        part_name=source_row.get("part_name", ""),
        image_name=source_row.get("image", ""),
        image_path=source_image_path,
        explicit_calib_path=args.calib_path.resolve() if args.calib_path else None,
    )
    if calib_path is None:
        raise FileNotFoundError(f"no calibration txt found for image {source_row.get('image', '')} under {cap_dir}")
    k = parse_projection_k(calib_path)

    source_t_bag = float(source_row["camera_t_bag"])
    source_nav = interp_nav(nav, source_t_bag)
    source_mode = "mat"
    source_desc = ""
    fz_start = int(float(source_row.get("mat_fz_start", source_row.get("pkt_start", 0)) or 0))
    fz_end = int(float(source_row.get("mat_fz_end", source_row.get("pkt_end", 0)) or 0))

    if source_row.get("mat"):
        mat_path = mat_dir / source_row["mat"]
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        source_desc = mat_path.name
        p_world, ranges, heights_abs, powers, beam_fzs, center_pose = build_source_world_points(
            bin_path=bin_path,
            mat_path=mat_path,
            fz_start=fz_start,
            fz_end=fz_end,
            source_t_bag=source_t_bag,
            nav=nav,
            yaw_deg=args.yaw,
        )
    elif source_row.get("ra"):
        ra_path = ra_dir / source_row["ra"]
        if not ra_path.exists():
            raise FileNotFoundError(ra_path)
        source_mode = "ra_fallback"
        source_desc = ra_path.name
        p_world, ranges, heights_abs, powers, beam_fzs, center_pose = build_source_world_points_from_rae(
            ra_path=ra_path,
            source_row=source_row,
            source_t_bag=source_t_bag,
            nav=nav,
            yaw_deg=args.yaw,
            threshold_percentile=args.ra_threshold_percentile,
            max_points=args.max_ra_points,
        )
    else:
        raise KeyError("aligned CSV row contains neither 'mat' nor 'ra'")

    print(f"source seq={args.source_seq}")
    print(f"  source image time: {source_t_bag:.3f}s")
    print(f"  match csv: {match_csv}")
    print(f"  nav csv: {nav_csv}")
    print(f"  bin path: {bin_path}")
    print(f"  source mode: {source_mode}")
    print(f"  source data: {source_desc}")
    print(f"  calib: {calib_path}")
    print(f"  raw FZ: {fz_start}..{fz_end}")
    print(f"  target seqs: {', '.join(r['camera_seq'] for r in target_rows)}")
    print(f"  source radar points after strongest-CFAR-per-beam selection: {len(p_world):,}")
    if len(p_world) == 0:
        raise RuntimeError("no radar points after filtering")

    overlay_paths: list[Path] = []
    sheet_titles: list[str] = []
    roi_rows: list[dict[str, object]] = []
    frame_dir = out_dir / f"source{args.source_seq:04d}"
    for row in target_rows:
        seq = int(row["camera_seq"])
        image_path = image_path_for_row(row, image_dir, match_csv)
        with Image.open(image_path) as image:
            image_size = image.size
        target_nav = interp_nav(nav, float(row["camera_t_bag"]))
        u, v, mask, depth = project_world_to_image(
            p_world=p_world,
            center_pose=center_pose,
            source_nav=source_nav,
            target_nav=target_nav,
            k=k,
            image_size=image_size,
            tilt_deg=args.tilt,
            use_pitch_roll=args.camera_attitude,
        )
        csv_path = frame_dir / "csv" / f"img{seq:04d}_radar_depth.csv"
        out_img = frame_dir / "vis" / f"img{seq:04d}_radar_overlay.png"
        write_depth_csv(csv_path, u, v, mask, depth, ranges, heights_abs, powers, beam_fzs)
        dt = float(row["camera_t_bag"]) - source_t_bag
        roi = visible_roi_from_projection(u, v, mask, image_size, pad_px=max(args.roi_pad, 0))
        if roi is not None:
            roi_x0, roi_y0, roi_x1, roi_y1, visible_points, coverage = roi
            roi_rows.append({
                "source_seq": args.source_seq,
                "camera_seq": seq,
                "image": row.get("image", ""),
                "image_path": str(image_path.resolve()),
                "camera_t_bag": float(row["camera_t_bag"]),
                "dt_sec": dt,
                "calib_path": str(calib_path.resolve()),
                "tilt_deg": args.tilt,
                "yaw_deg": args.yaw,
                "lever_x_m": args.lever_x,
                "lever_y_m": args.lever_y,
                "lever_z_m": args.lever_z,
                "roi_x0": roi_x0,
                "roi_y0": roi_y0,
                "roi_x1": roi_x1,
                "roi_y1": roi_y1,
                "visible_points": visible_points,
                "coverage_ratio": coverage,
            })
        title = (
            f"radar -> image seq#{seq}  dt={dt:+.3f}s  "
            f"pts={int(mask.sum())}  tilt={args.tilt:+.1f} yaw={args.yaw:+.1f}  "
            f"strongest CFAR / beam + sky filter"
        )
        draw_overlay(image_path, u, v, mask, depth, ranges, heights_abs, powers, title, out_img, args.label_limit)
        overlay_paths.append(out_img)
        sheet_titles.append(f"seq#{seq} dt={dt:+.3f}s pts={int(mask.sum())}")
        print(f"  seq#{seq}: points={int(mask.sum()):4d}  vis={out_img}  csv={csv_path}")

    roi_csv = args.roi_csv.resolve() if args.roi_csv else frame_dir / "projection_roi_constraints.csv"
    write_roi_constraints_csv(roi_csv, roi_rows)
    sheet_path = frame_dir / f"contact_sheet_win{args.time_window_s:g}s.png"
    draw_contact_sheet(overlay_paths, sheet_titles, sheet_path)
    print(f"ROI constraints: {roi_csv}")
    print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
