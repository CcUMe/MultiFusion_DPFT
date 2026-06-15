"""Run DPRT 2D inference on LH_all_sensor images and attach radar positions.

The structured recognition results do not contain image-plane coordinates.
Visual detections and radar candidates are therefore associated per class by
their horizontal order: image boxes are sorted left-to-right and radar
candidates are sorted by azimuth from left-to-right.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
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


def attach_positions(
    detections: List[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    names: Sequence[str],
    azimuth_order: str,
    min_radar_confidence: float,
) -> None:
    valid_candidates = []
    candidates_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if float(candidate.get("pred_confidence", 0.0)) >= min_radar_confidence:
            valid_candidates.append(candidate)
            candidates_by_label[str(candidate.get("pred_label", ""))].append(candidate)

    detections_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        detection["label"] = names[detection["label_index"]]
        detection["position_match"] = None
        detections_by_label[detection["label"]].append(detection)

    reverse = azimuth_order == "descending"
    for label, visual_group in detections_by_label.items():
        radar_group = candidates_by_label.get(label, [])
        visual_group.sort(key=lambda item: (item["box_xyxy"][0] + item["box_xyxy"][2]) * 0.5)
        radar_group.sort(key=candidate_azimuth, reverse=reverse)
        for detection, candidate in zip(visual_group, radar_group):
            detection["position_match"] = {
                "x_forward_m": float(candidate["x_forward_m"]),
                "y_left_m": float(candidate["y_left_m"]),
                "z_up_m": float(candidate["z_up_m"]),
            }
    if valid_candidates:
        nearest = min(valid_candidates, key=lambda candidate: float(candidate["geom_range_m"]))
        nearest_position = {
            "x_forward_m": float(nearest["x_forward_m"]),
            "y_left_m": float(nearest["y_left_m"]),
            "z_up_m": float(nearest["z_up_m"]),
        }
        for detection in detections:
            if detection["position_match"] is None:
                detection["position_match"] = nearest_position.copy()


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


def visualize(image_path: Path, detections: Sequence[Dict[str, Any]], output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        x1, x2 = int(round(x1 * width)), int(round(x2 * width))
        y1, y2 = int(round(y1 * height)), int(round(y2 * height))
        match = detection["position_match"]
        color = MATCHED_COLOR if match else UNMATCHED_COLOR
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        lines = [f"{detection['label']} {detection['score']:.2f}"]
        if match:
            lines.append(f"xyz=({match['x_forward_m']:.1f}, {match['y_left_m']:.1f}, {match['z_up_m']:.1f})m")
        else:
            lines.append("position: unmatched")
        draw_text_lines(image, lines, max(0, x1), max(18, y1), color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def iter_records(path: Path, limit: int) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if limit > 0 and index >= limit:
                break
            yield json.loads(line)



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
    parser.add_argument('--src', default='/mnt/disk1/yangqilin/dataset/LH_all_sensor/4_30/with_cameras_capture_20260430_101120')
    parser.add_argument('--cfg', default=None, help='Model config. Defaults to the config.json beside the checkpoint directory.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dst', default='/mnt/disk1/zhangzhibin/test/lh_all_sensor_positions')
    parser.add_argument("--draw-modality", choices=list(MODALITY_DIRS), default="camera_mono")
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    capture_dir = Path(args.src)
    image_results = capture_dir / "recognition_candidates" / "structured_results" / "image_results.jsonl"
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
    saved = 0

    with torch.no_grad():
        for record in iter_records(image_results, args.limit):
            visible_path = Path(record["image_path"])
            timestamp = float(record["camera_t_bag"])
            paths = {name: resolver.resolve(visible_path, name, timestamp) for name in active_inputs}
            if any(path is None for path in paths.values()):
                continue

            batch = to_device(prepare_batch(paths, config, homography), device)
            output = model(batch)
            detections = postprocess(output, args.score_threshold, args.nms_iou, 100)
            attach_positions(detections, record["detections"], names, "descending", 0.0)

            draw_path = paths[args.draw_modality]
            visualize(draw_path, detections, output_dir / draw_path.relative_to(capture_dir))
            saved += 1
            if saved % 100 == 0:
                print(f"saved {saved} images")

    print(f"Done. Saved {saved} images to {output_dir}")


if __name__ == "__main__":
    main()
