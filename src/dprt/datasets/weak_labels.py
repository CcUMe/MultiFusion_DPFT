from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


DEFAULT_CLASS_DEPTH_RATIOS = {
    0: 0.40,
    1: 0.65,
    2: 0.60,
    3: 0.60,
    4: 0.90,
    5: 1.00,
    6: 1.00,
}

DEFAULT_CLASS_BBOX_SCALES = {
    0: 1.05,
    1: 1.05,
    2: 1.10,
    3: 1.10,
    4: 1.15,
    5: 1.15,
    6: 1.15,
}


def _int_keyed(defaults: Dict[int, float], updates: Optional[Dict]) -> Dict[int, float]:
    values = dict(defaults)
    for key, value in (updates or {}).items():
        values[int(key)] = float(value)
    return values


def weak_label_config(config: Optional[Dict]) -> Dict[str, object]:
    config = config or {}
    global_depth = config.get("middle_slice_depth_ratio")
    if global_depth is None:
        class_depth_ratios = _int_keyed(DEFAULT_CLASS_DEPTH_RATIOS, config.get("class_depth_ratios"))
    else:
        class_depth_ratios = {class_id: float(global_depth) for class_id in DEFAULT_CLASS_DEPTH_RATIOS}
        class_depth_ratios = _int_keyed(class_depth_ratios, config.get("class_depth_ratios"))

    global_scale = config.get("bbox_scale")
    if global_scale is None:
        class_bbox_scales = _int_keyed(DEFAULT_CLASS_BBOX_SCALES, config.get("class_bbox_scales"))
    else:
        class_bbox_scales = {class_id: float(global_scale) for class_id in DEFAULT_CLASS_BBOX_SCALES}
        class_bbox_scales = _int_keyed(class_bbox_scales, config.get("class_bbox_scales"))

    return {
        "bbox_mode": config.get("bbox_mode", "middle_slice"),
        "line_mode": config.get("line_mode", "source_yaw"),
        "include_radar_labels": bool(config.get("include_radar_labels", False)),
        "class_depth_ratios": class_depth_ratios,
        "class_bbox_scales": class_bbox_scales,
    }


def length_height_panel_corners(
    center: np.ndarray,
    length_height: np.ndarray,
    yaw: np.ndarray,
    depth_ratios: np.ndarray,
) -> np.ndarray:
    """Return an image-facing length-height panel centered on each object.

    Weak image labels use object length as the image horizontal extent and
    height as the image vertical extent. In the processed K-Radar label frame,
    image horizontal motion is primarily the lateral y axis, while x is depth.
    Placing length on x would often project to a near-vertical line for distant
    objects, so this panel spans y-z at the object's center depth.
    """
    signs = np.asarray(
        [
            [0, -1, -1],
            [0, 1, -1],
            [0, 1, 1],
            [0, -1, 1],
        ],
        dtype=np.float64,
    )
    n = center.shape[0]
    length = length_height[:, 0] * np.clip(depth_ratios.reshape(-1), 0.0, 1.0)
    height = length_height[:, 1]
    panel_size = np.zeros((n, 3), dtype=np.float64)
    panel_size[:, 1] = length
    panel_size[:, 2] = height
    points = signs[None] * (panel_size[:, None, :] / 2.0)
    return points + center[:, None, :]


def _rotate_translate(points: np.ndarray, center: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    n = center.shape[0]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.zeros((n, 3, 3), dtype=np.float64)
    rot[:, 0, 0] = cos_yaw
    rot[:, 0, 1] = -sin_yaw
    rot[:, 1, 0] = sin_yaw
    rot[:, 1, 1] = cos_yaw
    rot[:, 2, 2] = 1.0
    return np.einsum("nij,nkj->nki", rot, points) + center[:, None, :]


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
    yaw: np.ndarray,
    depth_ratios: np.ndarray,
    bbox_scales: np.ndarray,
    projection: Optional[np.ndarray],
    image_shape: Optional[Tuple[int, int]],
    bbox_mode: str,
) -> np.ndarray:
    boxes = np.full((centers.shape[0], 4), np.nan, dtype=np.float32)
    if bbox_mode == "none" or projection is None or image_shape is None or centers.size == 0:
        return boxes

    points = length_height_panel_corners(centers, length_heights, yaw, depth_ratios)

    xy, valid = project_points(points, projection)
    img_h, img_w = image_shape

    for idx in range(centers.shape[0]):
        if not np.any(valid[idx]):
            continue
        pts = xy[idx, valid[idx]]
        x1, y1 = np.nanmin(pts, axis=0)
        x2, y2 = np.nanmax(pts, axis=0)
        x1 = np.clip(x1, 0, img_w - 1)
        x2 = np.clip(x2, 0, img_w - 1)
        y1 = np.clip(y1, 0, img_h - 1)
        y2 = np.clip(y2, 0, img_h - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        scale = float(bbox_scales[idx])
        if scale != 1.0:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            half_w = (x2 - x1) * scale / 2.0
            half_h = (y2 - y1) * scale / 2.0
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
    if centers.size == 0 or mode == "none":
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


def convert_raw_labels(
    raw_labels: np.ndarray,
    projection: Optional[np.ndarray],
    image_shape: Optional[Tuple[int, int]],
    num_classes: int,
    config: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    config = weak_label_config(config)
    raw_labels = np.asarray(raw_labels)
    if raw_labels.size == 0:
        raw_labels = raw_labels.reshape(0, 9)

    centers = raw_labels[:, 0:3].astype(np.float32)
    source_yaw = raw_labels[:, 3].astype(np.float32)
    length_heights = raw_labels[:, (4, 6)].astype(np.float32)
    class_ids = raw_labels[:, 7].astype(np.int64)

    class_depth_ratios = config["class_depth_ratios"]
    class_bbox_scales = config["class_bbox_scales"]
    depth_ratios = np.asarray([class_depth_ratios.get(int(c), 0.5) for c in class_ids], dtype=np.float32)
    bbox_scales = np.asarray([class_bbox_scales.get(int(c), 1.0) for c in class_ids], dtype=np.float32)

    bbox_mode = str(config["bbox_mode"])
    if bbox_mode in {"strict", "middle_slice"}:
        bbox_yaw = source_yaw
    elif bbox_mode in {"weak", "weak_middle_slice"}:
        bbox_yaw = np.zeros_like(source_yaw)
    else:
        bbox_yaw = np.zeros_like(source_yaw)

    image_boxes = projected_boxes(
        centers,
        length_heights,
        bbox_yaw,
        depth_ratios,
        bbox_scales,
        projection,
        image_shape,
        bbox_mode,
    )
    class_one_hot = np.zeros((class_ids.shape[0], num_classes), dtype=np.float32)
    valid = (class_ids + 1 >= 0) & (class_ids + 1 < num_classes)
    class_one_hot[np.arange(class_ids.shape[0])[valid], class_ids[valid] + 1] = 1.0

    label = {
        "gt_class": class_one_hot,
        "gt_center": centers,
        "gt_size_lh": length_heights,
        "gt_image_bbox": image_boxes,
        "source_yaw": source_yaw,
    }
    if config["include_radar_labels"]:
        label["gt_radar_center"] = centers[:, :2].astype(np.float32)
        label["gt_radar_line"] = radar_line_endpoints(
            centers, length_heights, source_yaw, str(config["line_mode"])
        )
    return label
