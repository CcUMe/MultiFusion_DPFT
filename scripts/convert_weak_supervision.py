import argparse
import json
import os
import os.path as osp
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_CATEGORY_NAMES = {
    0: "Sedan",
    1: "Bus or Truck",
    2: "Motorcycle",
    3: "Bicycle",
    4: "Bicycle Group",
    5: "Pedestrian",
    6: "Pedestrian Group",
}

DEFAULT_CLASS_DEPTH_RATIOS = {
    0: 0.40,  # Sedan
    1: 0.65,  # Bus or Truck
    2: 0.60,  # Motorcycle
    3: 0.60,  # Bicycle
    4: 0.90,  # Bicycle Group
    5: 1.00,  # Pedestrian
    6: 1.00,  # Pedestrian Group
}

DEFAULT_CLASS_BBOX_SCALES = {
    0: 1.05,  # Sedan
    1: 1.05,  # Bus or Truck
    2: 1.10,  # Motorcycle
    3: 1.10,  # Bicycle
    4: 1.15,  # Bicycle Group
    5: 1.15,  # Pedestrian
    6: 1.15,  # Pedestrian Group
}

COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 192, 255),
    (255, 0, 192),
    (128, 255, 0),
    (0, 128, 255),
    (255, 0, 0),
]


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for visualization. Install opencv-python or rerun with --no-visualize."
        ) from exc
    return cv2


def iter_sample_dirs(src: str, splits: Iterable[str]) -> Iterable[Tuple[str, str]]:
    for split in splits:
        split_dir = osp.join(src, split)
        if not osp.isdir(split_dir):
            continue
        for sequence in sorted(os.listdir(split_dir)):
            sequence_dir = osp.join(split_dir, sequence)
            if not osp.isdir(sequence_dir):
                continue
            for sample in sorted(os.listdir(sequence_dir)):
                sample_dir = osp.join(sequence_dir, sample)
                if osp.isdir(sample_dir) and osp.isfile(osp.join(sample_dir, "labels.npy")):
                    yield split, sample_dir


def load_projection_matrix(sample_dir: str) -> Optional[np.ndarray]:
    info_path = osp.join(sample_dir, "mono_info.npy")
    if not osp.isfile(info_path):
        return None

    matrix = np.load(info_path)
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.shape == (3, 4):
        return matrix
    if matrix.shape == (4, 4):
        return matrix[:3]
    if matrix.shape == (3, 3):
        return np.concatenate([matrix, np.zeros((3, 1), dtype=matrix.dtype)], axis=1)

    flat = matrix.reshape(-1)
    if flat.size == 12:
        return flat.reshape(3, 4)
    if flat.size == 16:
        return flat.reshape(4, 4)[:3]

    print(f"[WARN] Unsupported mono_info.npy shape {matrix.shape}: {info_path}")
    return None


def length_height_panel_corners(
    center: np.ndarray,
    length_height: np.ndarray,
    yaw: np.ndarray,
    depth_ratio: np.ndarray,
) -> np.ndarray:
    """Return an image-facing length-height panel centered on each object.

    Weak labels use object length as image horizontal extent and height as
    image vertical extent. The panel spans y-z at the object's center depth;
    using x as length would collapse distant objects into vertical lines.
    """
    n = center.shape[0]
    ratio = np.clip(np.asarray(depth_ratio, dtype=np.float64).reshape(n), 0.0, 1.0)
    signs = np.asarray(
        [
            [0, -1, -1],
            [0, 1, -1],
            [0, 1, 1],
            [0, -1, 1],
        ],
        dtype=np.float64,
    )
    panel_size = np.zeros((n, 3), dtype=np.float64)
    panel_size[:, 1] = length_height[:, 0] * ratio
    panel_size[:, 2] = length_height[:, 1]
    local = signs[None] * (panel_size[:, None, :] / 2.0)
    return local + center[:, None, :]


def project_points(points: np.ndarray, projection: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)], axis=1)
    projected = homogeneous @ projection.T
    depth = projected[:, 2]
    valid = depth > 1e-6
    xy = np.full((flat.shape[0], 2), np.nan, dtype=np.float64)
    xy[valid] = projected[valid, :2] / depth[valid, None]
    return xy.reshape(points.shape[:-1] + (2,)), valid.reshape(points.shape[:-1])


def projected_boxes(
    centers: np.ndarray,
    length_heights: np.ndarray,
    class_ids: np.ndarray,
    projection: Optional[np.ndarray],
    image_shape: Optional[Tuple[int, int]],
    yaw: np.ndarray,
    middle_slice_depth_ratios: np.ndarray,
    bbox_scales: np.ndarray,
) -> np.ndarray:
    boxes = np.full((centers.shape[0], 4), np.nan, dtype=np.float32)
    if projection is None or image_shape is None or centers.size == 0:
        return boxes

    points = length_height_panel_corners(centers, length_heights, yaw, middle_slice_depth_ratios)
    xy, valid = project_points(points, projection)
    img_h, img_w = image_shape

    for idx in range(centers.shape[0]):
        visible = valid[idx]
        if not np.any(visible):
            continue
        pts = xy[idx, visible]
        x1, y1 = np.nanmin(pts, axis=0)
        x2, y2 = np.nanmax(pts, axis=0)
        x1 = np.clip(x1, 0, img_w - 1)
        x2 = np.clip(x2, 0, img_w - 1)
        y1 = np.clip(y1, 0, img_h - 1)
        y2 = np.clip(y2, 0, img_h - 1)
        if x2 > x1 and y2 > y1:
            bbox_scale = float(bbox_scales[idx])
            if bbox_scale != 1.0:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                half_w = (x2 - x1) * bbox_scale / 2.0
                half_h = (y2 - y1) * bbox_scale / 2.0
                x1 = np.clip(cx - half_w, 0, img_w - 1)
                x2 = np.clip(cx + half_w, 0, img_w - 1)
                y1 = np.clip(cy - half_h, 0, img_h - 1)
                y2 = np.clip(cy + half_h, 0, img_h - 1)
            boxes[idx] = [x1, y1, x2, y2]

    return boxes


def radar_line_endpoints(
    centers: np.ndarray,
    length_heights: np.ndarray,
    yaw: np.ndarray,
    mode: str,
) -> np.ndarray:
    lines = np.full((centers.shape[0], 2, 2), np.nan, dtype=np.float32)
    if centers.size == 0:
        return lines

    if mode == "none":
        return lines
    if mode == "y_axis":
        yaw = np.full_like(yaw, np.pi / 2.0)
    elif mode == "x_axis":
        yaw = np.zeros_like(yaw)

    half_len = length_heights[:, 0] / 2.0
    direction = np.column_stack([np.cos(yaw), np.sin(yaw)])
    centers_xy = centers[:, :2]
    lines[:, 0, :] = centers_xy - direction * half_len[:, None]
    lines[:, 1, :] = centers_xy + direction * half_len[:, None]
    return lines


def convert_labels(
    raw_labels: np.ndarray,
    class_depth_ratios: Dict[int, float],
    class_bbox_scales: Dict[int, float],
    bbox_mode: str,
    line_mode: str,
    include_radar_labels: bool,
    projection: Optional[np.ndarray],
    image_shape: Optional[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    if raw_labels.size == 0:
        raw_labels = raw_labels.reshape(0, 9)

    centers = raw_labels[:, 0:3].astype(np.float32)
    original_yaw = raw_labels[:, 3].astype(np.float32)
    length_heights = raw_labels[:, (4, 6)].astype(np.float32)
    class_ids = raw_labels[:, 7].astype(np.int64)

    depth_ratios = np.asarray(
        [class_depth_ratios.get(int(class_id), 0.50) for class_id in class_ids],
        dtype=np.float32,
    )
    bbox_scales = np.asarray(
        [class_bbox_scales.get(int(class_id), 1.0) for class_id in class_ids],
        dtype=np.float32,
    )

    if bbox_mode in {"strict", "middle_slice"}:
        bbox_yaw = original_yaw
    elif bbox_mode in {"weak", "weak_middle_slice"}:
        bbox_yaw = np.zeros_like(original_yaw)
    else:
        bbox_yaw = np.zeros_like(original_yaw)

    if bbox_mode == "none":
        image_boxes = np.full((centers.shape[0], 4), np.nan, dtype=np.float32)
    else:
        image_boxes = projected_boxes(
            centers,
            length_heights,
            class_ids,
            projection,
            image_shape,
            bbox_yaw,
            depth_ratios,
            bbox_scales,
        )
    labels = {
        "class_id": class_ids,
        "center_xyz": centers,
        "size_lh": length_heights,
        "image_bbox_xyxy": image_boxes,
        "source_yaw": original_yaw,
    }
    if include_radar_labels:
        labels["radar_center_xy"] = centers[:, :2].astype(np.float32)
        labels["radar_line_xy"] = radar_line_endpoints(
            centers, length_heights, original_yaw, line_mode
        )
    return labels


def draw_image_overlay(
    image: np.ndarray,
    labels: Dict[str, np.ndarray],
    category_names: Dict[int, str],
) -> np.ndarray:
    cv2 = require_cv2()
    canvas = image.copy()
    for idx, box in enumerate(labels["image_bbox_xyxy"]):
        if not np.isfinite(box).all():
            continue
        class_id = int(labels["class_id"][idx])
        color = COLORS[class_id % len(COLORS)]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = category_names.get(class_id, str(class_id))
        cv2.putText(canvas, text, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return canvas


def draw_radar_overlay(
    labels: Dict[str, np.ndarray],
    category_names: Dict[int, str],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    canvas_size: Tuple[int, int],
) -> np.ndarray:
    cv2 = require_cv2()
    width, height = canvas_size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    def to_px(xy: np.ndarray) -> Tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        px = int((y - y_range[0]) / (y_range[1] - y_range[0]) * (width - 1))
        py = int((x_range[1] - x) / (x_range[1] - x_range[0]) * (height - 1))
        return px, py

    for x in np.linspace(x_range[0], x_range[1], 9):
        y0 = to_px(np.asarray([x, y_range[0]]))
        y1 = to_px(np.asarray([x, y_range[1]]))
        cv2.line(canvas, y0, y1, (225, 225, 225), 1)
    for y in np.linspace(y_range[0], y_range[1], 9):
        x0 = to_px(np.asarray([x_range[0], y]))
        x1 = to_px(np.asarray([x_range[1], y]))
        cv2.line(canvas, x0, x1, (225, 225, 225), 1)

    cv2.putText(canvas, "radar plane: x forward, y lateral", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)

    if "radar_center_xy" not in labels or "radar_line_xy" not in labels:
        return canvas

    for idx, center in enumerate(labels["radar_center_xy"]):
        class_id = int(labels["class_id"][idx])
        color = COLORS[class_id % len(COLORS)]
        p = to_px(center)
        line = labels["radar_line_xy"][idx]
        if np.isfinite(line).all():
            cv2.line(canvas, to_px(line[0]), to_px(line[1]), color, 2)
        cv2.circle(canvas, p, 4, color, -1)
        text = category_names.get(class_id, str(class_id))
        cv2.putText(canvas, text, (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return canvas


def save_npz(path: str, labels: Dict[str, np.ndarray]) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    np.savez_compressed(path, **labels)


def labels_summary(labels: Dict[str, np.ndarray], category_names: Dict[int, str]) -> Dict[str, object]:
    unique, counts = np.unique(labels["class_id"], return_counts=True)
    return {
        "num_objects": int(labels["class_id"].shape[0]),
        "classes": {
            category_names.get(int(class_id), str(int(class_id))): int(count)
            for class_id, count in zip(unique, counts)
        },
    }


def process_sample(
    sample_dir: str,
    output_root: str,
    src: str,
    category_names: Dict[int, str],
    class_depth_ratios: Dict[int, float],
    class_bbox_scales: Dict[int, float],
    bbox_mode: str,
    line_mode: str,
    include_radar_labels: bool,
    visualize: bool,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Dict[str, object]:
    cv2 = require_cv2() if visualize else None
    raw_labels = np.load(osp.join(sample_dir, "labels.npy"))
    image_path = osp.join(sample_dir, "mono.jpg")
    image = cv2.imread(image_path) if visualize and osp.isfile(image_path) else None
    image_shape = image.shape[:2] if image is not None else None
    projection = load_projection_matrix(sample_dir)

    labels = convert_labels(
        raw_labels,
        class_depth_ratios,
        class_bbox_scales,
        bbox_mode,
        line_mode,
        include_radar_labels,
        projection,
        image_shape,
    )

    rel_sample = osp.relpath(sample_dir, src)
    out_dir = osp.join(output_root, rel_sample)
    save_npz(osp.join(out_dir, "weak_labels.npz"), labels)

    summary = labels_summary(labels, category_names)
    with open(osp.join(out_dir, "weak_labels_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if visualize:
        vis_dir = osp.join(output_root, "visualizations", rel_sample.replace(os.sep, "__"))
        os.makedirs(vis_dir, exist_ok=True)
        if image is not None:
            cv2.imwrite(osp.join(vis_dir, "image_2d_boxes.jpg"), draw_image_overlay(image, labels, category_names))
        if include_radar_labels:
            cv2.imwrite(
                osp.join(vis_dir, "radar_plane.jpg"),
                draw_radar_overlay(labels, category_names, x_range, y_range, (900, 700)),
            )

    return {
        "sample": rel_sample,
        **summary,
    }


def parse_key_value_floats(values: Optional[List[str]], defaults: Dict[int, float]) -> Dict[int, float]:
    result = dict(defaults)
    for value in values or []:
        key, raw = value.split("=", 1)
        result[int(key)] = float(raw)
    return result


def build_class_depth_ratios(
    global_ratio: Optional[float],
    overrides: Optional[List[str]],
) -> Dict[int, float]:
    if global_ratio is None:
        defaults = DEFAULT_CLASS_DEPTH_RATIOS
    else:
        defaults = {class_id: float(global_ratio) for class_id in DEFAULT_CATEGORY_NAMES}
    return parse_key_value_floats(overrides, defaults)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert strict 3D labels.npy files into weak supervision labels and visual overlays."
    )
    parser.add_argument("--src", help="Processed dataset root containing train/val/test folders.", default="/mnt/disk1/zhangzhibin/dataset/kradar/")
    parser.add_argument("--dst", help="Output root for weak labels and visualizations.", default="/mnt/disk1/yangqilin/dataset/weak/kradar/")
    parser.add_argument("--splits", nargs="+", default=["val"], help="Dataset splits to scan.")
    # parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Dataset splits to scan.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick inspection.")
    parser.add_argument(
        "--bbox-mode",
        choices=["strict", "middle_slice", "weak", "weak_middle_slice", "none"],
        default="middle_slice",
        help=(
            "How to derive image 2D boxes. strict/middle_slice use source yaw; "
            "weak modes use yaw=0. All modes project a length-height panel."
        ),
    )
    parser.add_argument(
        "--middle-slice-depth-ratio",
        type=float,
        default=None,
        help=(
            "Global length fraction used by middle_slice modes. If omitted, class-specific defaults are used."
        ),
    )
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=None,
        help="Global projected image box scale. If omitted, class-specific defaults are used.",
    )
    parser.add_argument(
        "--class-depth-ratio",
        action="append",
        help="Override middle-slice length fraction per class, e.g. --class-depth-ratio 1=0.70.",
    )
    parser.add_argument(
        "--class-bbox-scale",
        action="append",
        help="Override projected image box scale per class, e.g. --class-bbox-scale 5=1.20.",
    )
    parser.add_argument(
        "--line-mode",
        choices=["source_yaw", "x_axis", "y_axis", "none"],
        default="source_yaw",
        help="How to derive radar plane line endpoints from center and length.",
    )
    parser.add_argument(
        "--include-radar-labels",
        action="store_true",
        help="Also save radar center/line labels and radar-plane visualizations.",
    )
    parser.add_argument("--no-visualize", action="store_true", help="Skip image/radar overlay outputs.")
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.0, 72.0], metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", nargs=2, type=float, default=[-6.4, 6.4], metavar=("MIN", "MAX"))
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    class_depth_ratios = build_class_depth_ratios(args.middle_slice_depth_ratio, args.class_depth_ratio)
    if args.bbox_scale is None:
        class_bbox_scale_defaults = DEFAULT_CLASS_BBOX_SCALES
    else:
        class_bbox_scale_defaults = {
            class_id: float(args.bbox_scale)
            for class_id in DEFAULT_CATEGORY_NAMES
        }
    class_bbox_scales = parse_key_value_floats(args.class_bbox_scale, class_bbox_scale_defaults)

    summaries = []
    for index, (_, sample_dir) in enumerate(iter_sample_dirs(args.src, args.splits)):
        if args.max_samples is not None and index >= args.max_samples:
            break
        summaries.append(
            process_sample(
                sample_dir=sample_dir,
                output_root=args.dst,
                src=args.src,
                category_names=DEFAULT_CATEGORY_NAMES,
                class_depth_ratios=class_depth_ratios,
                class_bbox_scales=class_bbox_scales,
                bbox_mode=args.bbox_mode,
                line_mode=args.line_mode,
                include_radar_labels=args.include_radar_labels,
                visualize=not args.no_visualize,
                x_range=tuple(args.x_range),
                y_range=tuple(args.y_range),
            )
        )

    summary_path = osp.join(args.dst, "conversion_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"num_samples": len(summaries), "samples": summaries}, f, indent=2)

    print(f"Converted {len(summaries)} samples.")
    print(f"Summary: {summary_path}")
    if not args.no_visualize:
        print(f"Visualizations: {osp.join(args.dst, 'visualizations')}")


if __name__ == "__main__":
    main()
