from __future__ import annotations

import json
import os
import os.path as osp
import re

from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from dprt.utils.config import get_active_inputs


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TIME_RE = re.compile(r"_t(?P<time>\d+(?:\.\d+)?)")


def parse_time(path: Path) -> float | None:
    match = TIME_RE.search(path.stem)
    if match is None:
        return None
    return float(match.group("time"))


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def build_time_index(image_dir: Path) -> List[Tuple[float, Path]]:
    indexed = []
    if not image_dir.exists():
        return indexed
    for image_path in image_dir.iterdir():
        if not is_image(image_path):
            continue
        timestamp = parse_time(image_path)
        if timestamp is not None:
            indexed.append((timestamp, image_path))
    indexed.sort(key=lambda item: item[0])
    return indexed


def nearest_by_time(index: List[Tuple[float, Path]], timestamp: float) -> Tuple[Path, float] | None:
    if not index:
        return None
    times = [item[0] for item in index]
    pos = bisect_left(times, timestamp)
    candidates = []
    if pos < len(index):
        candidates.append(index[pos])
    if pos > 0:
        candidates.append(index[pos - 1])
    if not candidates:
        return None
    nearest_time, nearest_path = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return nearest_path, abs(nearest_time - timestamp)


def shape_bbox(shape: Dict[str, Any]) -> Tuple[float, float, float, float] | None:
    points = shape.get("points") or []
    if len(points) < 2:
        return None
    xs, ys = [], []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        xs.append(float(point[0]))
        ys.append(float(point[1]))
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def project_bbox_by_homography(
    bbox: Tuple[float, float, float, float],
    homography: np.ndarray,
    target_w: int,
    target_h: int,
) -> Tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox
    pts = np.float32([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
    pts_ir = cv2.perspectiveTransform(pts, homography)
    xs = pts_ir[:, 0, 0]
    ys = pts_ir[:, 0, 1]
    nx1 = clamp(float(xs.min()), 0, target_w - 1)
    ny1 = clamp(float(ys.min()), 0, target_h - 1)
    nx2 = clamp(float(xs.max()), 0, target_w - 1)
    ny2 = clamp(float(ys.max()), 0, target_h - 1)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


def clamp_bbox_xyxy(
    bbox: Tuple[float, float, float, float],
    target_w: int,
    target_h: int,
) -> Tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox
    nx1 = clamp(float(x1), 0, target_w - 1)
    ny1 = clamp(float(y1), 0, target_h - 1)
    nx2 = clamp(float(x2), 0, target_w - 1)
    ny2 = clamp(float(y2), 0, target_h - 1)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


class LHPairsDataset(Dataset):
    def __init__(
        self,
        src: str,
        split: str = "train",
        manifest: str | None = None,
        homography: str = "/mnt/disk1/zhangzhibin/draw/H.npy",
        image_size: Union[int, Tuple[int, int]] = 512,
        visible_dir_name: str = "hikrobot_camera__DA8679037__image_raw",
        infrared_dir_name: str = "usb_ir__image_raw",
        micro_light_dir_name: str = "hikrobot_camera__DA8679038__image_raw",
        max_time_diff: float = 0.25,
        categories: Dict[str, int] | None = None,
        label_aliases: Dict[str, str] | None = None,
        inputs: List[str] | None = None,
        label_reference_input: str = "camera_mono",
        shared_label_inputs: List[str] | None = None,
        val_ratio: float = 0.2,
        dtype: str = "float32",
        **kwargs,
    ):
        super().__init__()
        self.src = src
        self.split = split
        self.manifest = manifest or osp.join(src, "scan_manifest.json")
        self.homography_path = homography
        self.homography = np.load(homography).astype(np.float64)
        self.inverse_homography = np.linalg.inv(self.homography)
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.visible_dir_name = visible_dir_name
        self.infrared_dir_name = infrared_dir_name
        self.micro_light_dir_name = micro_light_dir_name
        self.max_time_diff = max_time_diff
        self.val_ratio = val_ratio
        self.dtype = dtype
        self.inputs = inputs if inputs is not None else ["camera_mono", "ir_image"]
        self.label_reference_input = label_reference_input
        self.shared_label_inputs = set(shared_label_inputs or [])
        unsupported_inputs = set(self.inputs) - {"camera_mono", "ir_image", "micro_light"}
        if unsupported_inputs:
            raise ValueError(
                "LHPairsDataset currently supports only camera_mono, ir_image, and micro_light. "
                f"Please disable {sorted(unsupported_inputs)} in model.input_enable "
                "or extend the dataset loader with real modality data."
            )
        if self.label_reference_input not in {"camera_mono", "ir_image"}:
            raise ValueError(
                f"Unsupported label_reference_input={self.label_reference_input!r}. "
                "Expected 'camera_mono' or 'ir_image'."
            )
        unsupported_shared = self.shared_label_inputs - {"camera_mono", "ir_image", "micro_light"}
        if unsupported_shared:
            raise ValueError(f"Unsupported shared_label_inputs: {sorted(unsupported_shared)}")
        self.label_aliases = {"building complex": "Building complex", "Building comlplex": "Building complex"}
        if label_aliases:
            self.label_aliases.update(label_aliases)
        self.categories = self._normalize_categories(categories)
        self.samples = self._build_samples()

    @classmethod
    def from_config(cls, config: Dict[str, Any], *args, **kwargs) -> "LHPairsDataset":
        dataset_config = dict(config["computing"] | config["data"])
        dataset_config["inputs"] = get_active_inputs(config.get("model", {}))
        return cls(*args, **dataset_config, **kwargs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        sample = self.samples[index]
        target = self._load_target(sample)

        rgb_original_size = target.pop("_rgb_original_size")
        ir_original_size = target.pop("_ir_original_size")
        micro_original_size = target.pop("_micro_original_size")
        rgb_h, rgb_w = int(rgb_original_size[0].item()), int(rgb_original_size[1].item())
        ir_h, ir_w = int(ir_original_size[0].item()), int(ir_original_size[1].item())
        micro_h, micro_w = int(micro_original_size[0].item()), int(micro_original_size[1].item())

        inputs = {"rgb_original_size": rgb_original_size}
        if "camera_mono" in self.inputs:
            rgb = self._load_image(sample["rgb_image"])
            rgb, _ = self._resize_image(rgb)
            rgb = rgb.movedim(0, -1)
            inputs["camera_mono"] = rgb
            inputs["camera_mono_shape"] = torch.as_tensor(rgb.shape)

        if "ir_image" in self.inputs:
            ir = self._load_image(sample["ir_image"])
            ir, _ = self._resize_image(ir)
            ir = ir.movedim(0, -1)
            inputs["ir_image"] = ir
            inputs["ir_image_shape"] = torch.as_tensor(ir.shape)
            inputs["ir_original_size"] = ir_original_size
            inputs["ir_image_original_size"] = ir_original_size
            inputs["homography_rgb_to_ir_image"] = torch.from_numpy(self.homography).to(dtype=torch.float32)
            inputs["homography_rgb_to_ir"] = inputs["homography_rgb_to_ir_image"]

        if "micro_light" in self.inputs:
            micro = self._load_image(sample["micro_light"])
            micro, _ = self._resize_image(micro)
            micro = micro.movedim(0, -1)
            inputs["micro_light"] = micro
            inputs["micro_light_shape"] = torch.as_tensor(micro.shape)
            inputs["micro_light_original_size"] = micro_original_size
            inputs["homography_rgb_to_micro_light"] = torch.from_numpy(self.homography).to(dtype=torch.float32)

        if target["boxes"].numel():
            target["boxes"] = self._normalize_xyxy(target["boxes"], rgb_w, rgb_h)
            target["ir_boxes"] = self._normalize_xyxy(target["ir_boxes"], max(ir_w, 1), max(ir_h, 1))
            target["micro_boxes"] = self._normalize_xyxy(target["micro_boxes"], max(micro_w, 1), max(micro_h, 1))
            target["boxes_cxcywh"] = xyxy_to_cxcywh(target["boxes"])
            target["ir_boxes_cxcywh"] = xyxy_to_cxcywh(target["ir_boxes"])
            target["micro_boxes_cxcywh"] = xyxy_to_cxcywh(target["micro_boxes"])
        else:
            target["boxes_cxcywh"] = target["boxes"]
            target["ir_boxes_cxcywh"] = target["ir_boxes"]
            target["micro_boxes_cxcywh"] = target["micro_boxes"]

        if self.inputs == ["ir_image"]:
            target = self._filter_projected_only_target(target, "ir_valid")
            target["boxes"] = target["ir_boxes"]
            target["boxes_cxcywh"] = target["ir_boxes_cxcywh"]
        if self.inputs == ["micro_light"]:
            target = self._filter_projected_only_target(target, "micro_valid")
            target["boxes"] = target["micro_boxes"]
            target["boxes_cxcywh"] = target["micro_boxes_cxcywh"]

        target["image_id"] = torch.as_tensor(sample["index"], dtype=torch.long)
        return inputs, target

    @staticmethod
    def _filter_projected_only_target(target: Dict[str, torch.Tensor], valid_key: str) -> Dict[str, torch.Tensor]:
        valid = target[valid_key]
        return {
            key: value[valid]
            if value.ndim > 0 and value.shape[0] == valid.shape[0]
            else value
            for key, value in target.items()
        }

    def _normalize_categories(self, categories: Dict[str, int] | None) -> Dict[str, int]:
        if categories:
            normalized = {}
            for name, idx in categories.items():
                if int(idx) < 0:
                    continue
                normalized[name] = int(idx)
            if "Background" not in normalized:
                normalized = {"Background": 0, **normalized}
            return normalized

        labels = set()
        with open(self.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for entry in manifest:
            label_dir = Path(entry.get("visible_label_dir", ""))
            if not label_dir.exists():
                continue
            for json_path in label_dir.glob("*.json"):
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                for shape in data.get("shapes", []):
                    label = self.label_aliases.get(str(shape.get("label", "")), str(shape.get("label", "")))
                    if label:
                        labels.add(label)
        return {"Background": 0, **{label: idx for idx, label in enumerate(sorted(labels), start=1)}}


    def _build_prepared_samples(self) -> List[Dict[str, Any]] | None:
        split_dir = Path(self.src) / self.split
        if not split_dir.exists():
            return None

        sample_dirs = sorted(
            path for path in split_dir.glob("*/*/data_*")
            if (path / "label.json").exists()
            and ("camera_mono" not in self.inputs or (path / "visible.jpg").exists())
            and ("ir_image" not in self.inputs or (path / "infrared.jpg").exists())
            and ("micro_light" not in self.inputs or (path / "micro.jpg").exists())
        )
        if not sample_dirs:
            return None

        samples = []
        for sample_dir in sample_dirs:
            meta = {}
            meta_path = sample_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            rgb_image = sample_dir / "visible.jpg"
            ir_image = sample_dir / "infrared.jpg"
            micro_image = sample_dir / "micro.jpg"
            samples.append({
                "index": len(samples),
                "json": sample_dir / "label.json",
                "rgb_image": rgb_image if rgb_image.exists() else None,
                "ir_image": ir_image if ir_image.exists() else None,
                "micro_light": micro_image if micro_image.exists() else None,
                "meta": meta,
                "day": meta.get("day", sample_dir.parent.parent.name),
                "segment": meta.get("segment", ""),
            })
        return samples

    def _build_samples(self) -> List[Dict[str, Any]]:
        prepared_samples = self._build_prepared_samples()
        if prepared_samples is not None:
            return prepared_samples

        with open(self.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        segment_entries = []
        for entry in manifest:
            if not entry.get("has_visible_label_dir"):
                continue
            label_dir = Path(entry["visible_label_dir"])
            source_segment = Path(entry["segment_source"])
            rgb_dir = source_segment / "images" / self.visible_dir_name
            ir_dir = source_segment / "images" / self.infrared_dir_name
            micro_dir = source_segment / "images" / self.micro_light_dir_name
            if not label_dir.exists():
                continue
            if "camera_mono" in self.inputs and not rgb_dir.exists():
                continue
            if "ir_image" in self.inputs and (not entry.get("has_infrared_dir") or not ir_dir.exists()):
                continue
            if "micro_light" in self.inputs and not micro_dir.exists():
                continue
            segment_entries.append((entry, label_dir, rgb_dir, ir_dir, micro_dir))

        samples = []
        for entry, label_dir, rgb_dir, ir_dir, micro_dir in segment_entries:
            ir_index = build_time_index(ir_dir) if ir_dir.exists() else []
            micro_index = build_time_index(micro_dir) if micro_dir.exists() else []
            for json_path in sorted(label_dir.glob("*.json")):
                source_time = parse_time(json_path)
                if source_time is None:
                    continue
                rgb_image = self._find_source_image(json_path, rgb_dir) if rgb_dir.exists() else None
                if "camera_mono" in self.inputs and rgb_image is None:
                    continue

                ir_image = None
                if "ir_image" in self.inputs:
                    nearest = nearest_by_time(ir_index, source_time)
                    if nearest is None:
                        continue
                    ir_image, diff = nearest
                    if diff > self.max_time_diff:
                        continue
                elif ir_index:
                    nearest = nearest_by_time(ir_index, source_time)
                    if nearest is not None and nearest[1] <= self.max_time_diff:
                        ir_image = nearest[0]

                micro_image = None
                if "micro_light" in self.inputs:
                    nearest = nearest_by_time(micro_index, source_time)
                    if nearest is None:
                        continue
                    micro_image, diff = nearest
                    if diff > self.max_time_diff:
                        continue
                elif micro_index:
                    nearest = nearest_by_time(micro_index, source_time)
                    if nearest is not None and nearest[1] <= self.max_time_diff:
                        micro_image = nearest[0]

                samples.append({
                    "index": len(samples),
                    "json": json_path,
                    "rgb_image": rgb_image,
                    "ir_image": ir_image,
                    "micro_light": micro_image,
                    "day": entry.get("day", ""),
                    "segment": entry.get("segment", ""),
                })
        split_cut = int(round(len(samples) * (1.0 - self.val_ratio)))
        if self.split in {"val", "valid", "validation", "test"}:
            return samples[split_cut:]
        if self.split == "train":
            return samples[:split_cut]
        return samples

    @staticmethod
    def _find_source_image(json_path: Path, rgb_dir: Path) -> Path | None:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        image_path = data.get("imagePath")
        if image_path:
            candidate = rgb_dir / image_path
            if candidate.exists():
                return candidate
        for ext in IMAGE_EXTS:
            candidate = rgb_dir / f"{json_path.stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _load_image(self, path: Path) -> torch.Tensor:
        image = read_image(str(path)).type(getattr(torch, self.dtype)) / 255.0
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if image.shape[0] > 3:
            image = image[:3]
        return image

    def _resize_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[float, float]]:
        _, old_h, old_w = image.shape
        if self.image_size is None:
            return image, (1.0, 1.0)
        image = resize(image, self.image_size, antialias=True)
        _, new_h, new_w = image.shape
        return image, (new_w / old_w, new_h / old_h)

    @staticmethod
    def _normalize_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
        scale = torch.as_tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
        return torch.clamp(boxes / scale, min=0.0, max=1.0)

    def _load_target(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        with open(sample["json"], "r", encoding="utf-8") as f:
            data = json.load(f)

        rgb_w = int(data.get("imageWidth") or 0)
        rgb_h = int(data.get("imageHeight") or 0)
        if (rgb_h <= 0 or rgb_w <= 0) and sample.get("rgb_image") is not None:
            with torch.no_grad():
                rgb_image = read_image(str(sample["rgb_image"]))
            rgb_h, rgb_w = rgb_image.shape[-2:]

        ir_h, ir_w = 1, 1
        if sample.get("ir_image") is not None:
            with torch.no_grad():
                ir_image = read_image(str(sample["ir_image"]))
            ir_h, ir_w = ir_image.shape[-2:]

        micro_h, micro_w = 1, 1
        if sample.get("micro_light") is not None:
            with torch.no_grad():
                micro_image = read_image(str(sample["micro_light"]))
            micro_h, micro_w = micro_image.shape[-2:]

        boxes, ir_boxes, ir_valid, micro_boxes, micro_valid, labels = [], [], [], [], [], []
        for shape in data.get("shapes", []):
            label = str(shape.get("label", ""))
            label = self.label_aliases.get(label, label)
            class_idx = self.categories.get(label)
            if class_idx is None or class_idx <= 0:
                continue
            bbox = shape_bbox(shape)
            if bbox is None:
                continue

            if self.label_reference_input == "camera_mono":
                rgb_bbox = bbox
                ir_bbox = project_bbox_by_homography(bbox, self.homography, ir_w, ir_h)
                if "micro_light" in self.shared_label_inputs:
                    micro_bbox = clamp_bbox_xyxy(bbox, micro_w, micro_h)
                else:
                    micro_bbox = project_bbox_by_homography(bbox, self.homography, micro_w, micro_h)
            else:
                ir_bbox = bbox
                rgb_bbox = project_bbox_by_homography(bbox, self.inverse_homography, rgb_w, rgb_h)
                if "micro_light" in self.shared_label_inputs:
                    micro_bbox = clamp_bbox_xyxy(bbox, micro_w, micro_h)
                else:
                    micro_bbox = project_bbox_by_homography(bbox, self.inverse_homography, micro_w, micro_h)

            if rgb_bbox is None:
                boxes.append((0.0, 0.0, 0.0, 0.0))
            else:
                boxes.append(rgb_bbox)

            if ir_bbox is None:
                ir_boxes.append((0.0, 0.0, 0.0, 0.0))
                ir_valid.append(False)
            else:
                ir_boxes.append(ir_bbox)
                ir_valid.append(True)

            if micro_bbox is None:
                micro_boxes.append((0.0, 0.0, 0.0, 0.0))
                micro_valid.append(False)
            else:
                micro_boxes.append(micro_bbox)
                micro_valid.append(True)

            labels.append(class_idx)

        if not boxes:
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            return {
                "boxes": empty_boxes,
                "ir_boxes": empty_boxes.clone(),
                "ir_valid": torch.zeros((0,), dtype=torch.bool),
                "micro_boxes": empty_boxes.clone(),
                "micro_valid": torch.zeros((0,), dtype=torch.bool),
                "labels": torch.zeros((0,), dtype=torch.long),
                "_rgb_original_size": torch.as_tensor([rgb_h, rgb_w], dtype=torch.float32),
                "_ir_original_size": torch.as_tensor([ir_h, ir_w], dtype=torch.float32),
                "_micro_original_size": torch.as_tensor([micro_h, micro_w], dtype=torch.float32),
            }

        return {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "ir_boxes": torch.as_tensor(ir_boxes, dtype=torch.float32),
            "ir_valid": torch.as_tensor(ir_valid, dtype=torch.bool),
            "micro_boxes": torch.as_tensor(micro_boxes, dtype=torch.float32),
            "micro_valid": torch.as_tensor(micro_valid, dtype=torch.bool),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "_rgb_original_size": torch.as_tensor([rgb_h, rgb_w], dtype=torch.float32),
            "_ir_original_size": torch.as_tensor([ir_h, ir_w], dtype=torch.float32),
            "_micro_original_size": torch.as_tensor([micro_h, micro_w], dtype=torch.float32),
        }


def initialize_lh_pairs(*args: Any, **kwargs: Any) -> LHPairsDataset:
    return LHPairsDataset.from_config(*args, **kwargs)
