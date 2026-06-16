from __future__ import annotations

import argparse
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from common import load_csv_rows, power_to_db, safe_float, safe_int


PALETTE = {
    "Bridge": "#ffaa00",
    "Building complex": "#008cff",
    "Chimney": "#dc5050",
    "Power line": "#ffff00",
    "Power tower": "#ff0000",
    "Signal tower": "#b400ff",
    "Tall building": "#00c878",
    "Wind turbine": "#ffffff",
}

CAM_STABLE_SUBDIR = "hikrobot_camera__DA8679037__image_raw_stablelized_1920x1200"
CAM_RAW_SUBDIR = "hikrobot_camera__DA8679037__image_raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw radar recognition boxes and save side-by-side image+radar visualizations.")
    parser.add_argument("--prediction-csv", type=Path, required=True, help="Prediction CSV produced by rule_based_recognition.py")
    parser.add_argument("--out-dir", type=Path, help="Output directory. Defaults to <prediction-dir>/radar_overlays")
    parser.add_argument("--cap-dir", type=Path, help="Capture directory used to resolve the matched visible image for each scan")
    parser.add_argument("--mapping-csv", type=Path, help="Aligned mapping CSV. Defaults to <cap-dir>/image_to_antframe_time_aligned.csv")
    parser.add_argument("--min-confidence", type=float, default=0.18, help="Minimum confidence to draw a candidate")
    parser.add_argument("--topk-per-scan", type=int, default=12, help="Maximum drawn boxes per radar scan")
    parser.add_argument("--apply-camera-fov", action="store_true", help="Only draw radar targets that fall inside the visible camera FOV")
    parser.add_argument("--camera-hfov-deg", type=float, default=9.8, help="Visible camera horizontal field of view in degrees")
    parser.add_argument("--camera-vfov-deg", type=float, default=7.3, help="Visible camera vertical field of view in degrees")
    parser.add_argument("--camera-center-az-deg", type=float, default=0.0, help="Camera boresight azimuth in radar/body frame")
    parser.add_argument("--camera-center-el-deg", type=float, default=0.0, help="Camera boresight elevation in radar/body frame")
    return parser.parse_args()


def choose_layer_index(row: dict[str, str], n_layers: int) -> int:
    idx = safe_int(row.get("selected_pitch_idx"), 0)
    if 0 <= idx < n_layers:
        return idx
    return min(max(n_layers // 2, 0), max(0, n_layers - 1))


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        ra_path = row.get("ra_path", "")
        if ra_path:
            grouped[ra_path].append(row)
    return grouped


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


def row_in_camera_fov(
    row: dict[str, str],
    camera_hfov_deg: float,
    camera_vfov_deg: float,
    camera_center_az_deg: float,
    camera_center_el_deg: float,
) -> bool:
    az = safe_float(row.get("geom_azimuth_deg", row.get("azimuth_deg")), float("nan"))
    el = safe_float(row.get("geom_elevation_deg", row.get("elevation_deg", row.get("selected_pitch_deg"))), float("nan"))
    if not np.isfinite(az) or not np.isfinite(el):
        return False
    half_h = max(0.0, camera_hfov_deg) * 0.5
    half_v = max(0.0, camera_vfov_deg) * 0.5
    return abs(az - camera_center_az_deg) <= half_h and abs(el - camera_center_el_deg) <= half_v


def build_image_lookup(cap_dir: Path | None, mapping_csv: Path | None) -> dict[str, dict[str, str]]:
    if cap_dir is None or mapping_csv is None or not mapping_csv.exists():
        return {}
    rows = load_csv_rows(mapping_csv)
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("ra", "")
        if not key:
            continue
        if key not in lookup:
            lookup[key] = row
    return lookup


def render_radar_panel(
    ra_path: Path,
    rows: list[dict[str, str]],
    min_confidence: float,
    topk_per_scan: int,
    apply_camera_fov: bool,
    camera_hfov_deg: float,
    camera_vfov_deg: float,
    camera_center_az_deg: float,
    camera_center_el_deg: float,
) -> Image.Image | None:
    if not ra_path.exists():
        return None
    rows = [r for r in rows if safe_float(r.get("pred_confidence"), 0.0) >= min_confidence]
    if apply_camera_fov:
        rows = [
            r for r in rows
            if row_in_camera_fov(
                r,
                camera_hfov_deg=camera_hfov_deg,
                camera_vfov_deg=camera_vfov_deg,
                camera_center_az_deg=camera_center_az_deg,
                camera_center_el_deg=camera_center_el_deg,
            )
        ]
    if not rows:
        return None
    rows.sort(key=lambda r: safe_float(r.get("pred_confidence"), 0.0), reverse=True)
    rows = rows[:topk_per_scan]

    rae = np.load(ra_path, allow_pickle=False)
    if rae.ndim == 2:
        ra = rae
        layer_idx = 0
    elif rae.ndim == 3 and rae.shape[2] > 0:
        layer_idx = choose_layer_index(rows[0], rae.shape[2])
        ra = rae[:, :, layer_idx]
    else:
        return None

    ra_db = power_to_db(np.asarray(ra, dtype=np.float32))
    finite = np.isfinite(ra_db)
    if finite.any():
        vmin = float(np.percentile(ra_db[finite], 5.0))
        vmax = float(np.percentile(ra_db[finite], 99.5))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    range_max_m = safe_float(rows[0].get("range_max_m"), float(ra.shape[0]))
    az_min_deg = safe_float(rows[0].get("az_min_deg"), -60.0)
    az_max_deg = safe_float(rows[0].get("az_max_deg"), 60.0)

    fig, ax = plt.subplots(figsize=(7.2, 9.0), dpi=180)
    im = ax.imshow(
        ra_db,
        origin="lower",
        aspect="auto",
        extent=[az_min_deg, az_max_deg, 0.0, range_max_m],
        cmap="turbo",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    scan_idx = rows[0].get("scan_index", "?")
    ax.set_title(f"Radar Recognition Overlay | Scan {scan_idx} | Layer {layer_idx}", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("Power (dB)")

    for row in rows:
        center_az = safe_float(row.get("geom_azimuth_deg", row.get("azimuth_deg")), 0.0)
        center_range = safe_float(row.get("geom_range_m", row.get("range_m")), 0.0)
        span_az = safe_float(row.get("geom_az_span_deg", row.get("az_span_deg")), 0.0)
        span_range = safe_float(row.get("geom_range_span_m", row.get("range_span_m")), 0.0)

        x0_raw = safe_float(row.get("az_start_deg"), float("nan"))
        x1_raw = safe_float(row.get("az_end_deg"), float("nan"))
        y0_raw = safe_float(row.get("range_start_m"), float("nan"))
        y1_raw = safe_float(row.get("range_end_m"), float("nan"))

        if np.isfinite(x0_raw) and np.isfinite(x1_raw):
            left = min(x0_raw, x1_raw)
            right = max(x0_raw, x1_raw)
        else:
            left = center_az - 0.5 * span_az
            right = center_az + 0.5 * span_az

        if np.isfinite(y0_raw) and np.isfinite(y1_raw):
            bottom = min(y0_raw, y1_raw)
            top = max(y0_raw, y1_raw)
        else:
            bottom = center_range - 0.5 * span_range
            top = center_range + 0.5 * span_range
        width = max(1e-6, right - left)
        height = max(1e-6, top - bottom)
        label = row.get("pred_label", "Unknown")
        conf = safe_float(row.get("pred_confidence"), 0.0)
        xf = safe_float(row.get("x_forward_m"), 0.0)
        yl = safe_float(row.get("y_left_m"), 0.0)
        zu = safe_float(row.get("z_up_m"), 0.0)
        color = PALETTE.get(label, "#00ff00")
        rect = Rectangle((left, bottom), width, height, fill=False, edgecolor=color, linewidth=2.0)
        ax.add_patch(rect)
        text = f"{label} {conf:.2f}\nx={xf:.1f} y={yl:.1f} z={zu:.1f}"
        ax.text(
            left,
            min(range_max_m, top + 0.02 * max(range_max_m, 1.0)),
            text,
            fontsize=6,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"},
            path_effects=[matplotlib.patheffects.withStroke(linewidth=1.2, foreground="black")],
        )

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def compose_side_by_side(left_img: Image.Image | None, right_img: Image.Image) -> Image.Image:
    if left_img is None:
        return right_img
    target_h = max(left_img.height, right_img.height)
    if left_img.height != target_h:
        left_img = left_img.resize((int(round(left_img.width * target_h / left_img.height)), target_h))
    if right_img.height != target_h:
        right_img = right_img.resize((int(round(right_img.width * target_h / right_img.height)), target_h))
    canvas = Image.new("RGB", (left_img.width + right_img.width, target_h), (0, 0, 0))
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (left_img.width, 0))
    return canvas


def render_one_scan(
    ra_path: Path,
    rows: list[dict[str, str]],
    out_path: Path,
    min_confidence: float,
    topk_per_scan: int,
    image_lookup: dict[str, dict[str, str]],
    cap_dir: Path | None,
    apply_camera_fov: bool,
    camera_hfov_deg: float,
    camera_vfov_deg: float,
    camera_center_az_deg: float,
    camera_center_el_deg: float,
) -> bool:
    radar_panel = render_radar_panel(
        ra_path,
        rows,
        min_confidence,
        topk_per_scan,
        apply_camera_fov,
        camera_hfov_deg,
        camera_vfov_deg,
        camera_center_az_deg,
        camera_center_el_deg,
    )
    if radar_panel is None:
        return False

    left_img = None
    lookup_row = image_lookup.get(rows[0].get("ra", ""), {})
    if cap_dir is not None and lookup_row:
        image_path = find_image_path(cap_dir, lookup_row)
        if image_path is not None and image_path.exists():
            left_img = Image.open(image_path).convert("RGB")

    out_img = compose_side_by_side(left_img, radar_panel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    return True


def main() -> None:
    args = parse_args()
    pred_csv = args.prediction_csv.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else pred_csv.parent / "radar_overlays"
    cap_dir = args.cap_dir.resolve() if args.cap_dir else None
    mapping_csv = args.mapping_csv.resolve() if args.mapping_csv else (cap_dir / "image_to_antframe_time_aligned.csv" if cap_dir is not None else None)
    rows = load_csv_rows(pred_csv)
    if not rows:
        raise RuntimeError(f"no prediction rows found in {pred_csv}")

    image_lookup = build_image_lookup(cap_dir, mapping_csv)
    grouped = group_rows(rows)
    rendered = 0
    for ra_path_text, items in grouped.items():
        ra_path = Path(ra_path_text)
        stem = ra_path.stem
        out_path = out_dir / f"{stem}.png"
        ok = render_one_scan(
            ra_path,
            items,
            out_path,
            args.min_confidence,
            args.topk_per_scan,
            image_lookup,
            cap_dir,
            args.apply_camera_fov,
            args.camera_hfov_deg,
            args.camera_vfov_deg,
            args.camera_center_az_deg,
            args.camera_center_el_deg,
        )
        if ok:
            rendered += 1

    print(f"prediction csv: {pred_csv}")
    print(f"radar overlay dir: {out_dir}")
    print(f"rendered radar overlays: {rendered}")


if __name__ == "__main__":
    main()
