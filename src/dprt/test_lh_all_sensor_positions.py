"""Run DPRT 2D inference on LH_all_sensor images and attach radar positions.

The structured recognition results do not contain image-plane coordinates.
Visual detections and radar candidates are therefore associated per class by
their horizontal order: image boxes are sorted left-to-right and radar
candidates are sorted by azimuth from left-to-right.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.io import read_image
from torchvision.ops import nms
from torchvision.transforms.functional import resize


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dprt.models import build as build_model  # noqa: E402
from dprt.utils.config import get_active_inputs, load_config  # noqa: E402
from dprt.utils.misc import set_seed  # noqa: E402


TIME_RE = re.compile(r"_t(?P<time>\d+(?:\.\d+)?)")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODALITY_DIRS = {
    "camera_mono": "hikrobot_camera__DA8679037__image_raw",
    "ir_image": "usb_ir__image_raw",
    "micro_light": "hikrobot_camera__DA8679038__image_raw",
}
MATCHED_COLOR = (0, 210, 0)
UNMATCHED_COLOR = (0, 165, 255)
HEATMAP_COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "hot": cv2.COLORMAP_HOT,
}


def parse_time(path: Path) -> float | None:
    match = TIME_RE.search(path.stem)
    return float(match.group("time")) if match else None


def load_image(path: Path, image_size: int | Sequence[int], dtype: str) -> torch.Tensor:
    image = read_image(str(path)).type(getattr(torch, dtype)) / 255.0
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.shape[0] > 3:
        image = image[:3]
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    image = resize(image, size, antialias=True)
    return image.movedim(0, -1)


def build_time_index(directory: Path) -> List[Tuple[float, Path]]:
    if not directory.exists():
        return []
    indexed = []
    for path in directory.iterdir():
        timestamp = parse_time(path)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and timestamp is not None:
            indexed.append((timestamp, path))
    return sorted(indexed)


def nearest_by_time(index: Sequence[Tuple[float, Path]], timestamp: float) -> Tuple[Path, float] | None:
    if not index:
        return None
    nearest_time, nearest_path = min(index, key=lambda item: abs(item[0] - timestamp))
    return nearest_path, abs(nearest_time - timestamp)


class ModalityResolver:
    def __init__(self, max_time_diff: float):
        self.max_time_diff = max_time_diff
        self.cache: Dict[Path, List[Tuple[float, Path]]] = {}

    def resolve(self, visible_path: Path, input_name: str, timestamp: float) -> Path | None:
        if input_name == "camera_mono":
            return visible_path if visible_path.exists() else None
        image_root = visible_path.parent.parent
        directory = image_root / MODALITY_DIRS[input_name]
        if directory not in self.cache:
            self.cache[directory] = build_time_index(directory)
        nearest = nearest_by_time(self.cache[directory], timestamp)
        if nearest is None or nearest[1] > self.max_time_diff:
            return None
        return nearest[0]


def class_names(categories: Dict[str, int]) -> List[str]:
    names = ["Background"] * (max(int(value) for value in categories.values()) + 1)
    for name, index in categories.items():
        names[int(index)] = name
    return names


def to_device(data: Any, device: torch.device) -> Any:
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    return data


def prepare_batch(
    paths: Dict[str, Path],
    config: Dict[str, Any],
    homography: np.ndarray,
) -> Dict[str, torch.Tensor]:
    data_config = config["data"]
    dtype = config["computing"].get("dtype", "float32")
    image_size = data_config.get("image_size", 512)
    batch: Dict[str, torch.Tensor] = {}
    original_sizes: Dict[str, torch.Tensor] = {}

    for input_name, path in paths.items():
        image = read_image(str(path))
        original_sizes[input_name] = torch.tensor(image.shape[-2:], dtype=torch.float32)
        tensor = load_image(path, image_size, dtype)
        batch[input_name] = tensor.unsqueeze(0)
        batch[f"{input_name}_shape"] = torch.as_tensor(tensor.shape).unsqueeze(0)

    rgb_size = original_sizes.get("camera_mono")
    if rgb_size is None:
        # The current RGB/IR fuser uses RGB coordinates as its reference plane.
        rgb_size = next(iter(original_sizes.values()))
    batch["rgb_original_size"] = rgb_size.unsqueeze(0)

    homography_tensor = torch.from_numpy(homography.astype(np.float32)).unsqueeze(0)
    for input_name in paths:
        if input_name == "camera_mono":
            continue
        batch[f"{input_name}_original_size"] = original_sizes[input_name].unsqueeze(0)
        batch[f"homography_rgb_to_{input_name}"] = homography_tensor
    return batch


def postprocess(
    output: Dict[str, torch.Tensor],
    score_threshold: float,
    nms_iou: float,
    max_detections: int,
) -> List[Dict[str, Any]]:
    class_scores = output["class"][0]
    boxes = output["boxes_xyxy"][0]
    scores, labels = class_scores[:, 1:].max(dim=-1)
    labels = labels + 1
    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    selected = []
    for class_idx in labels.unique(sorted=True):
        local = torch.nonzero(labels == class_idx, as_tuple=False).flatten()
        if nms_iou > 0:
            local = local[nms(boxes[local], scores[local], nms_iou)]
        selected.append(local)
    if not selected:
        return []

    indices = torch.cat(selected)
    indices = indices[scores[indices].argsort(descending=True)[:max_detections]]
    return [
        {
            "box_xyxy": [float(value) for value in boxes[index].detach().cpu()],
            "score": float(scores[index].detach().cpu()),
            "label_index": int(labels[index].detach().cpu()),
        }
        for index in indices
    ]


def candidate_azimuth(candidate: Dict[str, Any]) -> float:
    if candidate.get("geom_azimuth_deg") is not None:
        return float(candidate["geom_azimuth_deg"])
    return math.degrees(math.atan2(float(candidate["y_left_m"]), float(candidate["x_forward_m"])))


def candidate_range(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("geom_range_m", float("inf")))


def detection_center_x(detection: Dict[str, Any]) -> float:
    x1, _, x2, _ = detection["box_xyxy"]
    return 0.5 * (x1 + x2)


def detection_bottom_y(detection: Dict[str, Any]) -> float:
    return float(detection["box_xyxy"][3])


def detection_area(detection: Dict[str, Any]) -> float:
    x1, y1, x2, y2 = detection["box_xyxy"]
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def normalize_values(values: Sequence[float], invert: bool = False) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        normalized = [0.5] * len(values)
    else:
        normalized = [(value - lo) / (hi - lo) for value in values]
    if invert:
        normalized = [1.0 - value for value in normalized]
    return normalized


def assign_detection_positions(
    detections: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    reverse: bool,
    same_label_only: bool,
) -> List[tuple[int, int]]:
    if not detections or not candidates:
        return []

    det_x = [detection_center_x(item) for item in detections]
    det_bottom = [detection_bottom_y(item) for item in detections]
    det_area = [detection_area(item) for item in detections]
    det_x_norm = normalize_values(det_x)
    det_bottom_norm = normalize_values(det_bottom, invert=True)
    det_area_norm = normalize_values(det_area, invert=True)
    det_far_prior = [0.5 * (bottom + area) for bottom, area in zip(det_bottom_norm, det_area_norm)]

    cand_az = [candidate_azimuth(item) for item in candidates]
    cand_range = [candidate_range(item) for item in candidates]
    cand_x_norm = normalize_values(cand_az, invert=reverse)
    cand_far_prior = normalize_values(cand_range)
    cand_conf_penalty = [1.0 - float(item.get("pred_confidence", 0.0)) for item in candidates]

    cost = np.zeros((len(detections), len(candidates)), dtype=np.float32)
    for i, detection in enumerate(detections):
        for j, candidate in enumerate(candidates):
            class_penalty = 0.0 if detection["label"] == str(candidate.get("pred_label", "")) else 1.25
            if same_label_only and class_penalty > 0.0:
                cost[i, j] = 1e6
                continue
            horizontal_cost = abs(det_x_norm[i] - cand_x_norm[j])
            depth_cost = abs(det_far_prior[i] - cand_far_prior[j])
            confidence_cost = cand_conf_penalty[j]
            cost[i, j] = 3.0 * horizontal_cost + 1.5 * depth_cost + 0.15 * confidence_cost + class_penalty

    row_ind, col_ind = linear_sum_assignment(cost)
    return [
        (int(row_idx), int(col_idx))
        for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist())
        if np.isfinite(cost[row_idx, col_idx]) and cost[row_idx, col_idx] < 1e5
    ]


def candidate_position(candidate: Dict[str, Any]) -> Dict[str, float]:
    return {
        "x_forward_m": float(candidate["x_forward_m"]),
        "y_left_m": float(candidate["y_left_m"]),
        "z_up_m": float(candidate["z_up_m"]),
    }


def average_candidate_position(candidates: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not candidates:
        raise ValueError('average_candidate_position requires at least one candidate.')
    return {
        "x_forward_m": float(sum(float(candidate["x_forward_m"]) for candidate in candidates) / len(candidates)),
        "y_left_m": float(sum(float(candidate["y_left_m"]) for candidate in candidates) / len(candidates)),
        "z_up_m": float(sum(float(candidate["z_up_m"]) for candidate in candidates) / len(candidates)),
    }


def attach_building_complex_positions(
    visual_group: Sequence[Dict[str, Any]],
    tall_building_detections: Sequence[Dict[str, Any]],
) -> set[int]:
    if not visual_group or not tall_building_detections:
        return set()

    matched_rows = set()
    for row_idx, detection in enumerate(visual_group):
        x1, _, x2, _ = detection["box_xyxy"]
        grouped_positions = [
            tall_detection["position_match"]
            for tall_detection in tall_building_detections
            if tall_detection.get("position_match") is not None
            and x1 <= detection_center_x(tall_detection) <= x2
        ]
        if not grouped_positions:
            continue
        visual_group[row_idx]["position_match"] = {
            "x_forward_m": float(sum(item["x_forward_m"] for item in grouped_positions) / len(grouped_positions)),
            "y_left_m": float(sum(item["y_left_m"] for item in grouped_positions) / len(grouped_positions)),
            "z_up_m": float(sum(item["z_up_m"] for item in grouped_positions) / len(grouped_positions)),
        }
        visual_group[row_idx]["position_source"] = "tall_building_mean"
        matched_rows.add(row_idx)
    return matched_rows


def _set_visual_priors(detections: Sequence[Dict[str, Any]]) -> None:
    centers = [detection_center_x(item) for item in detections]
    bottoms = [detection_bottom_y(item) for item in detections]
    areas = [detection_area(item) for item in detections]
    center_norm = normalize_values(centers)
    bottom_far = normalize_values(bottoms, invert=True)
    area_far = normalize_values(areas, invert=True)
    for detection, x_norm, bottom_score, area_score in zip(detections, center_norm, bottom_far, area_far):
        detection["_visual_x_norm"] = x_norm
        detection["_visual_far_prior"] = 0.5 * (bottom_score + area_score)


def estimate_position_from_anchors(
    detection: Dict[str, Any],
    anchors: Sequence[Dict[str, Any]],
) -> Dict[str, float] | None:
    if not anchors:
        return None

    target_x = float(detection.get("_visual_x_norm", 0.5))
    target_far = float(detection.get("_visual_far_prior", 0.5))
    weighted = []
    total_weight = 0.0
    for anchor in anchors:
        match = anchor.get("position_match")
        if match is None:
            continue
        anchor_x = float(anchor.get("_visual_x_norm", 0.5))
        anchor_far = float(anchor.get("_visual_far_prior", 0.5))
        dx = abs(target_x - anchor_x)
        df = abs(target_far - anchor_far)
        weight = 1.0 / (0.05 + 2.5 * dx + 1.5 * df)
        theta = math.atan2(float(match["y_left_m"]), max(float(match["x_forward_m"]), 1e-3))
        weighted.append((weight, match, theta))
        total_weight += weight

    if total_weight <= 0.0:
        return None

    x_forward = sum(weight * float(match["x_forward_m"]) for weight, match, _ in weighted) / total_weight
    theta = sum(weight * angle for weight, _, angle in weighted) / total_weight
    z_up = sum(weight * float(match["z_up_m"]) for weight, match, _ in weighted) / total_weight
    return {
        "x_forward_m": float(x_forward),
        "y_left_m": float(x_forward * math.tan(theta)),
        "z_up_m": float(z_up),
    }


def attach_tall_building_positions(
    visual_group: Sequence[Dict[str, Any]],
    radar_group: Sequence[Dict[str, Any]],
    reverse: bool,
    anchors: Sequence[Dict[str, Any]],
) -> set[int]:
    matched_rows = set()
    raw_positions: Dict[int, Dict[str, float]] = {}
    if visual_group and radar_group:
        assignments = assign_detection_positions(visual_group, radar_group, reverse, same_label_only=True)
        for row_idx, col_idx in assignments:
            raw_positions[row_idx] = candidate_position(radar_group[col_idx])

    for row_idx, detection in enumerate(visual_group):
        estimated = estimate_position_from_anchors(detection, anchors)
        if estimated is not None:
            detection["position_match"] = estimated
            detection["position_source"] = "visual_anchor_estimate"
            matched_rows.add(row_idx)
            continue
        if row_idx in raw_positions:
            detection["position_match"] = raw_positions[row_idx]
            detection["position_source"] = "same_label_raw"
            matched_rows.add(row_idx)
    return matched_rows


def attach_positions(
    detections: List[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    names: Sequence[str],
    azimuth_order: str,
    min_radar_confidence: float,
) -> None:
    valid_candidates = [
        candidate for candidate in candidates
        if float(candidate.get("pred_confidence", 0.0)) >= min_radar_confidence
    ]
    candidates_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in valid_candidates:
        candidates_by_label[str(candidate.get("pred_label", ""))].append(candidate)

    detections_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        detection["label"] = names[detection["label_index"]]
        detection["position_match"] = None
        detection["position_source"] = None
        detections_by_label[detection["label"]].append(detection)

    _set_visual_priors(detections)
    reverse = azimuth_order == "descending"
    unmatched = []

    for label, visual_group in detections_by_label.items():
        if label in {"Tall building", "Building complex"}:
            continue
        radar_group = candidates_by_label.get(label, [])
        assignments = assign_detection_positions(visual_group, radar_group, reverse, same_label_only=True)
        matched_rows = set()
        for row_idx, col_idx in assignments:
            visual_group[row_idx]["position_match"] = candidate_position(radar_group[col_idx])
            visual_group[row_idx]["position_source"] = "same_label"
            matched_rows.add(row_idx)
        for row_idx, detection in enumerate(visual_group):
            if row_idx not in matched_rows:
                unmatched.append(detection)

    anchor_detections = [
        detection
        for detection in detections
        if detection.get("position_match") is not None
        and detection["label"] not in {"Tall building", "Building complex"}
    ]

    tall_group = detections_by_label.get("Tall building", [])
    tall_matched = attach_tall_building_positions(
        tall_group,
        candidates_by_label.get("Tall building", []),
        reverse,
        anchor_detections,
    )
    for row_idx, detection in enumerate(tall_group):
        if row_idx not in tall_matched:
            unmatched.append(detection)

    building_group = detections_by_label.get("Building complex", [])
    building_matched = attach_building_complex_positions(building_group, tall_group)
    for row_idx, detection in enumerate(building_group):
        if row_idx not in building_matched:
            estimated = estimate_position_from_anchors(detection, anchor_detections)
            if estimated is not None:
                detection["position_match"] = estimated
                detection["position_source"] = "visual_anchor_estimate"
            else:
                unmatched.append(detection)

    if not valid_candidates:
        return

    if unmatched:
        assignments = assign_detection_positions(unmatched, valid_candidates, reverse, same_label_only=False)
        matched_rows = set()
        for row_idx, col_idx in assignments:
            unmatched[row_idx]["position_match"] = candidate_position(valid_candidates[col_idx])
            unmatched[row_idx]["position_source"] = "global_cost"
            matched_rows.add(row_idx)

        for row_idx, detection in enumerate(unmatched):
            if row_idx in matched_rows or detection.get("position_match") is not None:
                continue
            best_candidate = min(
                valid_candidates,
                key=lambda candidate: (
                    0.0 if detection["label"] == str(candidate.get("pred_label", "")) else 1.0,
                    abs(detection_center_x(detection) - candidate_azimuth(candidate) / 180.0),
                    candidate_range(candidate),
                )
            )
            detection["position_match"] = candidate_position(best_candidate)
            detection["position_source"] = "global_fallback"


def draw_text_lines(
    image: np.ndarray,
    lines: Sequence[str],
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    line_height = 18
    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    width = max(widths, default=0) + 8
    top = max(0, y - line_height * len(lines) - 4)
    right = min(image.shape[1] - 1, x + width)
    cv2.rectangle(image, (x, top), (right, y), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        text_y = top + 14 + index * line_height
        cv2.putText(image, line, (x + 4, text_y), font, scale, color, thickness, cv2.LINE_AA)


def detection_distance_m(detection: Dict[str, Any]) -> float | None:
    match = detection.get("position_match")
    if not match:
        return None
    return math.sqrt(
        float(match["x_forward_m"]) ** 2
        + float(match["y_left_m"]) ** 2
        + float(match["z_up_m"]) ** 2
    )


def detection_heat_value(
    detection: Dict[str, Any],
    heatmap_value_source: str,
    min_distance: float,
    max_distance: float,
) -> float:
    if heatmap_value_source == "score":
        return float(detection["score"])

    distance = detection_distance_m(detection)
    if distance is None:
        return float(detection["score"])
    if max_distance <= min_distance:
        return 1.0
    normalized = (distance - min_distance) / (max_distance - min_distance)
    return float(np.clip(1.0 - normalized, 0.0, 1.0))


def distance_intensity_bounds(detections: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    distances = [distance for distance in (detection_distance_m(d) for d in detections) if distance is not None]
    min_distance = min(distances) if distances else 0.0
    max_distance = max(distances) if distances else 0.0
    return min_distance, max_distance



def build_detection_heatmap(
    image_shape: Tuple[int, int],
    detections: Sequence[Dict[str, Any]],
    sigma_scale: float,
    heatmap_value_source: str,
) -> np.ndarray:
    height, width = image_shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    min_distance, max_distance = distance_intensity_bounds(detections)

    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        x1 = max(0, min(width - 1, int(round(x1 * width))))
        x2 = max(0, min(width - 1, int(round(x2 * width))))
        y1 = max(0, min(height - 1, int(round(y1 * height))))
        y2 = max(0, min(height - 1, int(round(y2 * height))))
        if x2 <= x1 or y2 <= y1:
            continue

        patch_w = x2 - x1 + 1
        patch_h = y2 - y1 + 1
        center_x = (patch_w - 1) * 0.5
        center_y = (patch_h - 1) * 0.5
        sigma_x = max(1.0, patch_w * sigma_scale)
        sigma_y = max(1.0, patch_h * sigma_scale)

        xs = np.arange(patch_w, dtype=np.float32)
        ys = np.arange(patch_h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        gaussian = np.exp(
            -(
                ((grid_x - center_x) ** 2) / (2.0 * sigma_x * sigma_x)
                + ((grid_y - center_y) ** 2) / (2.0 * sigma_y * sigma_y)
            )
        )
        heatmap[y1 : y2 + 1, x1 : x2 + 1] += gaussian * detection_heat_value(
            detection,
            heatmap_value_source,
            min_distance,
            max_distance,
        )

    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap



def overlay_distance_attention(
    image: np.ndarray,
    detections: Sequence[Dict[str, Any]],
) -> np.ndarray:
    base = image.astype(np.float32)
    height, width = image.shape[:2]
    min_distance, max_distance = distance_intensity_bounds(detections)

    color_field = np.zeros((height, width, 3), dtype=np.float32)
    alpha_field = np.zeros((height, width), dtype=np.float32)

    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        x1 = max(0, min(width - 1, int(round(x1 * width))))
        x2 = max(0, min(width - 1, int(round(x2 * width))))
        y1 = max(0, min(height - 1, int(round(y1 * height))))
        y2 = max(0, min(height - 1, int(round(y2 * height))))
        if x2 <= x1 or y2 <= y1:
            continue

        intensity = detection_heat_value(detection, "distance", min_distance, max_distance)
        peak_alpha = 0.14 + 0.42 * intensity

        box_w = x2 - x1 + 1
        box_h = y2 - y1 + 1
        center_x = x1 + box_w // 2
        center_y = y1 + box_h // 2
        radius_x = max(4, int(round(box_w * 0.48)))
        radius_y = max(4, int(round(box_h * 0.48)))

        left = max(0, center_x - radius_x)
        right = min(width - 1, center_x + radius_x)
        top = max(0, center_y - radius_y)
        bottom = min(height - 1, center_y + radius_y)
        if right <= left or bottom <= top:
            continue

        xs = np.arange(left, right + 1, dtype=np.float32)
        ys = np.arange(top, bottom + 1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        norm_x = (grid_x - center_x) / max(float(radius_x), 1.0)
        norm_y = (grid_y - center_y) / max(float(radius_y), 1.0)
        radial = norm_x * norm_x + norm_y * norm_y

        # Attention-like blob: smooth center focus with soft elliptical fade.
        strength = np.exp(-2.8 * radial)
        strength[radial > 1.35] = 0.0
        if not np.any(strength > 0):
            continue

        local_alpha = peak_alpha * strength
        heat_uint8 = np.clip((0.25 + 0.75 * strength) * 255.0, 0, 255).astype(np.uint8)
        local_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO).astype(np.float32)

        patch_alpha = alpha_field[top : bottom + 1, left : right + 1]
        patch_color = color_field[top : bottom + 1, left : right + 1]
        update_mask = local_alpha > patch_alpha
        patch_alpha[update_mask] = local_alpha[update_mask]
        patch_color[update_mask] = local_color[update_mask]

    nonzero = alpha_field > 0
    if not np.any(nonzero):
        return image

    overlay = base.copy()
    alpha = alpha_field[..., None]
    overlay[nonzero] = base[nonzero] * (1.0 - alpha[nonzero]) + color_field[nonzero] * alpha[nonzero]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
    threshold: float,
    colormap_name: str,
) -> np.ndarray:
    overlay = image.copy()
    mask = heatmap >= threshold
    if not np.any(mask):
        return overlay

    colored = cv2.applyColorMap(
        np.clip(heatmap * 255.0, 0, 255).astype(np.uint8),
        HEATMAP_COLORMAPS[colormap_name],
    )
    blended = cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0.0)
    overlay[mask] = blended[mask]
    return overlay


def visualize(
    image_path: Path,
    detections: Sequence[Dict[str, Any]],
    output_path: Path,
    visualization_mode: str,
    heatmap_alpha: float,
    heatmap_threshold: float,
    heatmap_sigma_scale: float,
    heatmap_colormap: str,
    heatmap_value_source: str,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    draw_boxes = visualization_mode in {"boxes", "both"}
    draw_heatmap = visualization_mode in {"heatmap", "both"}

    if draw_heatmap:
        if heatmap_value_source == "distance":
            image = overlay_distance_attention(image, detections)
        else:
            heatmap = build_detection_heatmap((height, width), detections, heatmap_sigma_scale, heatmap_value_source)
            image = overlay_heatmap(image, heatmap, heatmap_alpha, heatmap_threshold, heatmap_colormap)

    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        x1, x2 = int(round(x1 * width)), int(round(x2 * width))
        y1, y2 = int(round(y1 * height)), int(round(y2 * height))
        match = detection["position_match"]
        color = MATCHED_COLOR if match else UNMATCHED_COLOR
        if draw_boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        lines = [f"{detection['label']} {detection['score']:.2f}"]
        if match:
            lines.append(f"xyz=({match['x_forward_m']:.1f}, {match['y_left_m']:.1f}, {match['z_up_m']:.1f})m")
        else:
            lines.append("position: unmatched")

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        ellipse_radius_y = max(1, int(round((y2 - y1 + 1) * 0.32)))
        text_x = max(0, center_x - 70)
        text_y = max(22, center_y - ellipse_radius_y - 6)
        draw_text_lines(image, lines, text_x, text_y, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def discover_image_result_files(src: Path) -> List[Path]:
    direct = src / "recognition_candidates" / "structured_results" / "image_results.jsonl"
    if direct.is_file():
        return [direct]
    files = sorted(src.glob("**/recognition_candidates/structured_results/image_results.jsonl"))
    return [file for file in files if file.is_file()]



def iter_records(path: Path, limit: int) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if limit > 0 and index >= limit:
                break
            yield json.loads(line)



def load_records(paths: Sequence[Path], limit: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        remaining = 0 if limit <= 0 else max(limit - len(records), 0)
        if limit > 0 and remaining == 0:
            break
        records.extend(iter_records(path, remaining))
    return records



def record_scene_group_key(visible_path: Path, capture_dir: Path, group_level: str) -> str:
    if group_level == "segment":
        group_root = visible_path.parent.parent.parent
    elif group_level == "part":
        group_root = visible_path.parent.parent.parent.parent
    elif group_level == "with_dir":
        group_root = visible_path.parent.parent.parent.parent.parent
    else:
        raise ValueError(f"Unsupported group level: {group_level}")
    try:
        relative_root = group_root.relative_to(capture_dir)
    except ValueError:
        return str(group_root)
    return str(relative_root) if relative_root.parts else group_root.name



def sample_rank(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest, 16)



def build_visualization_selection(
    records: Sequence[Dict[str, Any]],
    capture_dir: Path,
    visualize_percent: float,
    group_level: str,
) -> set[str] | None:
    if visualize_percent >= 100.0:
        return None
    if visualize_percent <= 0.0:
        return set()

    grouped: Dict[str, List[tuple[int, str]]] = defaultdict(list)
    for record in records:
        visible_path = Path(record["image_path"])
        scene_key = record_scene_group_key(visible_path, capture_dir, group_level)
        path_str = str(visible_path)
        grouped[scene_key].append((sample_rank(path_str), path_str))

    selected: set[str] = set()
    for scene_items in grouped.values():
        scene_items.sort(key=lambda item: item[0])
        keep_count = math.ceil(len(scene_items) * visualize_percent / 100.0)
        for _, path_str in scene_items[:keep_count]:
            selected.add(path_str)
    return selected


def load_compatible_model(checkpoint: str, config: Dict[str, Any]) -> torch.nn.Module:
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_data["model_state_dict"]
    state_dict = {
        key.replace("mirco_light", "micro_light"): value
        for key, value in state_dict.items()
    }
    model = build_model(config["model"]["name"], config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    active_inputs = set(get_active_inputs(config["model"]))
    inactive_inputs = set(config["model"].get("inputs", [])) - active_inputs
    allowed_prefixes = tuple(
        prefix
        for input_name in inactive_inputs
        for prefix in (
            f"backbones.{input_name}.",
            f"necks.{input_name}.",
            f"embeddings.{input_name}.",
        )
    )
    invalid_missing = [key for key in missing_keys if not key.startswith(allowed_prefixes)]
    if invalid_missing or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing active keys: {invalid_missing[:20]}; "
            f"unexpected keys: {unexpected_keys[:20]}"
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser("Draw DPRT detections and positions on LH_all_sensor images")
    parser.add_argument('--src', default='/mnt/disk1/yangqilin/dataset/LH_all_sensor/')
    parser.add_argument('--cfg', default=None, help='Model config. Defaults to the config.json beside the checkpoint directory.')
    parser.add_argument('--checkpoint', default="/mnt/disk1/zhangzhibin/test/light/20260614-115352-428/checkpoints/20260614-115352-428_checkpoint_0102.pt", required=True)
    parser.add_argument('--dst', default='/mnt/disk1/yangqilin/dpft/lh_all_sensor_positions')
    parser.add_argument("--draw-modality", choices=list(MODALITY_DIRS), default="camera_mono")
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--visualization-mode", choices=["boxes", "heatmap", "both"], default="boxes")
    parser.add_argument("--heatmap-alpha", type=float, default=0.55)
    parser.add_argument("--heatmap-threshold", type=float, default=0.05)
    parser.add_argument("--heatmap-sigma-scale", type=float, default=0.25)
    parser.add_argument("--heatmap-colormap", choices=sorted(HEATMAP_COLORMAPS), default="turbo")
    parser.add_argument("--heatmap-value-source", choices=["score", "distance"], default="score")
    parser.add_argument("--visualize-percent", type=float, default=20.0)
    parser.add_argument("--visualize-group-level", choices=["with_dir", "part", "segment"], default="with_dir")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    capture_dir = Path(args.src)
    image_result_files = discover_image_result_files(capture_dir)
    if not image_result_files:
        raise FileNotFoundError(f"No image_results.jsonl found under {capture_dir}")
    output_dir = Path(args.dst)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.cfg) if args.cfg else Path(args.checkpoint).parent.parent / "config.json"
    config = load_config(str(config_path))
    config_text = json.dumps(config).replace("mirco_light", "micro_light")
    config = json.loads(config_text)
    print(f"Using config: {config_path}")
    set_seed(config["computing"]["seed"])
    if args.device:
        config["computing"]["device"] = args.device
    device = torch.device(config["computing"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    active_inputs = get_active_inputs(config["model"])
    unsupported = set(active_inputs) - set(MODALITY_DIRS)
    if unsupported:
        raise ValueError(f"This script supports image inputs only; disable: {sorted(unsupported)}")
    if args.draw_modality not in active_inputs:
        raise ValueError(f"--draw-modality {args.draw_modality!r} is not active: {active_inputs}")

    language_model = config.get("model", {}).get("language_model")
    if language_model is not None:
        language_model["enabled"] = False
    model = load_compatible_model(args.checkpoint, config)
    model.to(device).eval()

    homography = np.load(config["data"]["homography"])
    resolver = ModalityResolver(0.25)
    names = class_names(config["data"]["categories"])
    if not 0.0 <= args.visualize_percent <= 100.0:
        raise ValueError(f"--visualize-percent must be in [0, 100], got {args.visualize_percent}")

    print(f"Found {len(image_result_files)} image_results.jsonl files under {capture_dir}")
    records = load_records(image_result_files, args.limit)
    selected_paths = build_visualization_selection(records, capture_dir, args.visualize_percent, args.visualize_group_level)
    saved = 0

    with torch.no_grad():
        for record in records:
            visible_path = Path(record["image_path"])
            if selected_paths is not None and str(visible_path) not in selected_paths:
                continue
            timestamp = float(record["camera_t_bag"])
            paths = {name: resolver.resolve(visible_path, name, timestamp) for name in active_inputs}
            if any(path is None for path in paths.values()):
                continue

            batch = to_device(prepare_batch(paths, config, homography), device)
            output = model(batch)
            detections = postprocess(output, args.score_threshold, args.nms_iou, 100)
            attach_positions(detections, record["detections"], names, "descending", 0.0)

            draw_path = paths[args.draw_modality]
            visualize(
                draw_path,
                detections,

                output_dir / draw_path.relative_to(capture_dir),
                visualization_mode=args.visualization_mode,
                heatmap_alpha=args.heatmap_alpha,
                heatmap_threshold=args.heatmap_threshold,
                heatmap_sigma_scale=args.heatmap_sigma_scale,
                heatmap_colormap=args.heatmap_colormap,
                heatmap_value_source=args.heatmap_value_source,
            )
            saved += 1
            if saved % 100 == 0:
                print(f"saved {saved} images")

    print(f"Done. Saved {saved} images to {output_dir} (visualize_percent={args.visualize_percent:.2f}, visualize_group_level={args.visualize_group_level})")


if __name__ == "__main__":
    main()
