"""Visualize low-altitude mmWave RAE tensors next to matched camera images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import build_image_aligned_mmwave_mat as bridge
import match_radar_camera_anchor as anchor_match
from radar_config import DEFAULT_CONFIG_PATH, load_config


CAM_STABLE_SUBDIR = "hikrobot_camera__DA8679037__image_raw_stablelized_1920x1200"
CAM_RAW_SUBDIR = "hikrobot_camera__DA8679037__image_raw"
PKT = anchor_match.PKT
N_RANGE_OUT = 666


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render low-altitude radar heatmaps and save them side-by-side with matched images."
    )
    parser.add_argument("--cap-dir", type=Path, required=True, help="Capture directory containing the raw bin and *_part folders.")
    parser.add_argument("--mapping-csv", type=Path, help="image_to_scan_time_aligned.csv path.")
    parser.add_argument("--ra-dir", type=Path, help="Directory containing exported ra.npy files.")
    parser.add_argument("--preview-dir", type=Path, help="Output directory for radar and side-by-side previews.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Radar processing config JSON path.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional limit on processed mapping rows. 0 means all.")
    return parser.parse_args()


def load_mapping_rows(path: Path, max_rows: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_rows > 0:
        rows = rows[:max_rows]
    return rows


def select_one_image_per_ra(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        ra_name = row.get("ra", "")
        if not ra_name:
            continue
        grouped.setdefault(ra_name, []).append(row)

    selected: list[dict[str, str]] = []
    for items in grouped.values():
        def score(row: dict[str, str]) -> tuple[float, float, str]:
            center = float(row.get("scan_t_center", row.get("camera_t_bag", "0")) or 0.0)
            cam_t = float(row.get("camera_t_bag", "0") or 0.0)
            return (abs(cam_t - center), cam_t, row.get("image", ""))
        selected.append(min(items, key=score))

    selected.sort(key=lambda row: (float(row.get("scan_t_center", row.get("camera_t_bag", "0")) or 0.0), row.get("ra", "")))
    return selected


def find_image_path(cap_dir: Path, row: dict[str, str]) -> Path | None:
    part_dir = cap_dir / row["part_name"]
    candidates = [
        part_dir / "images" / CAM_STABLE_SUBDIR / row["image"],
        part_dir / "images" / CAM_RAW_SUBDIR / row["image"],
    ]

    if part_dir.exists():
        for seg_dir in sorted(part_dir.iterdir()):
            if not seg_dir.is_dir() or not seg_dir.name.startswith("segment_"):
                continue
            candidates.extend([
                seg_dir / "images" / CAM_STABLE_SUBDIR / row["image"],
                seg_dir / "images" / CAM_RAW_SUBDIR / row["image"],
            ])

    for path in candidates:
        if path.exists():
            return path
    return None


def read_frame_meta(bin_path: Path, pkt_start: int, pkt_end: int) -> dict[str, float]:
    n_pkts = pkt_end - pkt_start + 1
    with bin_path.open("rb") as f:
        f.seek(pkt_start * PKT)
        buf = f.read(n_pkts * PKT)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n_pkts, PKT)
    ant_az = arr[:, 80:84].copy().view("<f4").ravel().astype(np.float64)
    range_km = arr[:, 32:36].copy().view("<f4").ravel().astype(np.float64)

    valid_az = ant_az[np.isfinite(ant_az)]
    valid_range = range_km[np.isfinite(range_km)]
    return {
        "range_max_m": float(np.median(valid_range) * 1000.0) if len(valid_range) else float(N_RANGE_OUT),
        "az_min_deg": float(np.min(valid_az)) if len(valid_az) else -53.0,
        "az_max_deg": float(np.max(valid_az)) if len(valid_az) else 53.0,
    }


def meta_from_row_or_bin(row: dict[str, str], bin_path: Path) -> dict[str, float | int | list[float]]:
    has_csv_meta = row.get("range_max_m") and row.get("az_min_deg") and row.get("az_max_deg")
    if has_csv_meta:
        meta: dict[str, float | int | list[float]] = {
            "range_max_m": float(row["range_max_m"]),
            "az_min_deg": float(row["az_min_deg"]),
            "az_max_deg": float(row["az_max_deg"]),
        }
    else:
        meta = read_frame_meta(bin_path, int(row["pkt_start"]), int(row["pkt_end"]))

    if row.get("selected_pitch_deg"):
        meta["selected_pitch_deg"] = float(row["selected_pitch_deg"])
    if row.get("selected_pitch_idx"):
        meta["selected_pitch_idx"] = int(row["selected_pitch_idx"])
    if row.get("pitch_layer_count"):
        meta["pitch_layer_count"] = int(row["pitch_layer_count"])
    if row.get("pitch_layers_deg"):
        meta["pitch_layers_deg"] = [float(v) for v in row["pitch_layers_deg"].split(",") if v]
    if row.get("scan_dir"):
        meta["scan_dir"] = int(float(row["scan_dir"]))
    return meta


def choose_visualization_layer(ra: np.ndarray, meta: dict[str, float | int | list[float]], vis_cfg: dict) -> tuple[np.ndarray, int]:
    if ra.ndim != 3:
        return np.asarray(ra), 0

    n_layers = ra.shape[2]
    strategy = str(vis_cfg.get("layer_selection", "selected_pitch"))
    fallback = str(vis_cfg.get("layer_fallback", "middle"))

    if strategy == "fixed_index":
        layer_idx = int(vis_cfg.get("fixed_layer_index", 0))
    elif strategy == "fixed_pitch_deg":
        pitch_layers = meta.get("pitch_layers_deg")
        target_pitch = float(vis_cfg.get("fixed_pitch_deg", 0.0))
        if isinstance(pitch_layers, list) and len(pitch_layers) > 0:
            layer_idx = min(range(len(pitch_layers)), key=lambda i: abs(float(pitch_layers[i]) - target_pitch))
        else:
            layer_idx = -1
    elif strategy == "middle":
        layer_idx = n_layers // 2
    else:
        layer_idx = int(meta.get("selected_pitch_idx", -1))

    if layer_idx < 0 or layer_idx >= n_layers:
        if fallback == "first":
            layer_idx = 0
        elif fallback == "last":
            layer_idx = n_layers - 1
        else:
            layer_idx = n_layers // 2

    return ra[:, :, layer_idx], layer_idx


def render_ra_png(ra_source, out_path: Path, meta: dict[str, float | int | list[float]], vis_cfg: dict) -> None:
    if isinstance(ra_source, (str, Path)):
        ra = np.load(ra_source, allow_pickle=False)
    else:
        ra = np.asarray(ra_source)

    ra_vis, layer_idx = choose_visualization_layer(ra, meta, vis_cfg)
    meta["visualized_pitch_idx"] = int(layer_idx)
    pitch_layers = meta.get("pitch_layers_deg")
    if isinstance(pitch_layers, list) and 0 <= layer_idx < len(pitch_layers):
        meta["visualized_pitch_deg"] = float(pitch_layers[layer_idx])
    else:
        meta["visualized_pitch_deg"] = float(meta.get("selected_pitch_deg", 0.0))

    ra_vis = 20 * np.log10(np.maximum(np.asarray(ra_vis, dtype=np.float32), 1e-10))
    finite = np.isfinite(ra_vis)
    if finite.any():
        vmin = float(np.percentile(ra_vis[finite], 5.0))
        vmax = float(np.percentile(ra_vis[finite], 99.5))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(7.2, 9.0), dpi=180)
    im = ax.imshow(
        ra_vis,
        origin="lower",
        aspect="auto",
        extent=[meta["az_min_deg"], meta["az_max_deg"], 0.0, meta["range_max_m"]],
        cmap="turbo",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Low-Altitude mmWave RA Slice", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("Power (dB)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fit_to_height(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    new_w = int(round(img.width * target_h / img.height))
    return img.resize((new_w, target_h))


def compose_side_by_side(
    image_path: Path,
    radar_png_path: Path,
    out_path: Path,
    row: dict[str, str],
    meta: dict[str, float | int | list[float]],
    vis_cfg: dict,
) -> None:
    img = Image.open(image_path).convert("RGB")
    radar = Image.open(radar_png_path).convert("RGB")

    target_h = min(img.height, radar.height)
    img = fit_to_height(img, target_h)
    radar = fit_to_height(radar, target_h)

    gap = int(vis_cfg.get("side_by_side_gap_px", 24))
    margin = int(vis_cfg.get("side_by_side_margin_px", 16))
    label_h = int(vis_cfg.get("side_by_side_label_height_px", 72))
    canvas_w = img.width + radar.width + gap + margin * 2
    canvas = Image.new("RGB", (canvas_w, target_h + label_h), color=(255, 255, 255))
    img_x = margin
    radar_x = img_x + img.width + gap
    canvas.paste(img, (img_x, label_h))
    canvas.paste(radar, (radar_x, label_h))

    draw = ImageDraw.Draw(canvas)
    left_text = f"Image: {row['image']}"
    pitch_deg = float(meta.get("visualized_pitch_deg", meta.get("selected_pitch_deg", 0.0)))
    pitch_idx = int(meta.get("visualized_pitch_idx", meta.get("selected_pitch_idx", 0)))
    pitch_count = int(meta.get("pitch_layer_count", 1))
    right_text = (
        f"Radar: {row['ra']} | Scan {row.get('scan_index', '?')} | Layer {pitch_idx + 1}/{pitch_count} | "
        f"Pitch {pitch_deg:+.1f} deg | Range 0-{float(meta['range_max_m']):.1f} m | "
        f"Az {float(meta['az_min_deg']):.1f}..{float(meta['az_max_deg']):.1f} deg"
    )
    draw.text((img_x, 14), left_text, fill=(0, 0, 0))
    draw.text((radar_x, 14), right_text, fill=(0, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_cfg = cfg["ra_output"]
    vis_cfg = cfg["visualization"]

    cap_dir = args.cap_dir.resolve()
    mapping_csv = args.mapping_csv.resolve() if args.mapping_csv else cap_dir / output_cfg["mapping_csv_name"]
    ra_dir = args.ra_dir.resolve() if args.ra_dir else cap_dir / output_cfg["ra_dir_name"]
    preview_dir = args.preview_dir.resolve() if args.preview_dir else cap_dir / vis_cfg["preview_dir_name"]
    radar_subdir = preview_dir / vis_cfg["radar_subdir_name"]
    compare_subdir = preview_dir / vis_cfg["compare_subdir_name"]
    bin_path = bridge.resolve_bin_path(cap_dir, None)

    rows = load_mapping_rows(mapping_csv, args.max_rows)
    if not rows:
        raise RuntimeError(f"no rows found in {mapping_csv}")

    rows = select_one_image_per_ra(rows)

    rendered_radar = 0
    rendered_pairs = 0
    missing_images = 0
    missing_ra = 0

    for row in rows:
        ra_name = row.get("ra", "")
        if not ra_name:
            missing_ra += 1
            continue
        ra_path = ra_dir / ra_name
        if not ra_path.exists():
            missing_ra += 1
            continue

        image_path = find_image_path(cap_dir, row)
        if image_path is None:
            missing_images += 1
            continue

        meta = meta_from_row_or_bin(row, bin_path)
        radar_png = radar_subdir / f"{Path(ra_name).stem}.png"
        if not radar_png.exists():
            render_ra_png(ra_path, radar_png, meta, vis_cfg)
            rendered_radar += 1

        compare_png = compare_subdir / f"{Path(ra_name).stem}__{Path(row['image']).stem}.png"
        compose_side_by_side(image_path, radar_png, compare_png, row, meta, vis_cfg)
        rendered_pairs += 1

    print(f"config: {args.config.resolve()}")
    print(f"mapping csv: {mapping_csv}")
    print(f"bin: {bin_path}")
    print(f"ra dir: {ra_dir}")
    print(f"preview dir: {preview_dir}")
    print(f"selected full radar frames: {len(rows)}")
    print(f"full-frame radar previews rendered: {rendered_radar}")
    print(f"rendered side-by-side pairs: {rendered_pairs}")
    print(f"rows missing image: {missing_images}")
    print(f"rows missing ra: {missing_ra}")


if __name__ == "__main__":
    main()
