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
        max_time_diff: float = 0.25,
        categories: Dict[str, int] | None = None,
        label_aliases: Dict[str, str] | None = None,
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
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.visible_dir_name = visible_dir_name
        self.infrared_dir_name = infrared_dir_name
        self.max_time_diff = max_time_diff
        self.val_ratio = val_ratio
        self.dtype = dtype
        self.label_aliases = {"building complex": "Building complex"}
        if label_aliases:
            self.label_aliases.update(label_aliases)
        self.categories = self._normalize_categories(categories)
        self.samples = self._build_samples()

    @classmethod
    def from_config(cls, config: Dict[str, Any], *args, **kwargs) -> "LHPairsDataset":
        return cls(*args, **dict(config["computing"] | config["data"]), **kwargs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        sample = self.samples[index]
        rgb = self._load_image(sample["rgb_image"])
        ir = self._load_image(sample["ir_image"])

        target = self._load_target(sample)
        rgb_h, rgb_w = rgb.shape[-2:]
        ir_h, ir_w = ir.shape[-2:]

        rgb_original_size = torch.as_tensor([rgb_h, rgb_w], dtype=torch.float32)
        ir_original_size = torch.as_tensor([ir_h, ir_w], dtype=torch.float32)

        rgb, rgb_scale = self._resize_image(rgb)
        ir, ir_scale = self._resize_image(ir)

        inputs = {
            "camera_mono": rgb.movedim(0, -1),
            "ir_image": ir.movedim(0, -1),
            "camera_mono_shape": torch.as_tensor(rgb.movedim(0, -1).shape),
            "ir_image_shape": torch.as_tensor(ir.movedim(0, -1).shape),
            "rgb_original_size": rgb_original_size,
            "ir_original_size": ir_original_size,
            "homography_rgb_to_ir": torch.from_numpy(self.homography).to(dtype=torch.float32),
        }

        if target["boxes"].numel():
            rgb_boxes = target["boxes"] * torch.as_tensor(
                [rgb_scale[0], rgb_scale[1], rgb_scale[0], rgb_scale[1]],
                dtype=torch.float32,
            )
            ir_boxes = target["ir_boxes"] * torch.as_tensor(
                [ir_scale[0], ir_scale[1], ir_scale[0], ir_scale[1]],
                dtype=torch.float32,
            )
            target["boxes"] = self._normalize_xyxy(rgb_boxes, rgb.shape[-1], rgb.shape[-2])
            target["ir_boxes"] = self._normalize_xyxy(ir_boxes, ir.shape[-1], ir.shape[-2])
            target["boxes_cxcywh"] = xyxy_to_cxcywh(target["boxes"])
            target["ir_boxes_cxcywh"] = xyxy_to_cxcywh(target["ir_boxes"])
        else:
            target["boxes_cxcywh"] = target["boxes"]
            target["ir_boxes_cxcywh"] = target["ir_boxes"]

        target["image_id"] = torch.as_tensor(sample["index"], dtype=torch.long)
        return inputs, target

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
            if (path / "visible.jpg").exists()
            and (path / "infrared.jpg").exists()
            and (path / "label.json").exists()
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
            samples.append({
                "index": len(samples),
                "json": sample_dir / "label.json",
                "rgb_image": sample_dir / "visible.jpg",
                "ir_image": sample_dir / "infrared.jpg",
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
            if not entry.get("has_visible_label_dir") or not entry.get("has_infrared_dir"):
                continue
            label_dir = Path(entry["visible_label_dir"])
            source_segment = Path(entry["segment_source"])
            rgb_dir = source_segment / "images" / self.visible_dir_name
            ir_dir = source_segment / "images" / self.infrared_dir_name
            if not label_dir.exists() or not rgb_dir.exists() or not ir_dir.exists():
                continue
            segment_entries.append((entry, label_dir, rgb_dir, ir_dir))

        samples = []
        for entry, label_dir, rgb_dir, ir_dir in segment_entries:
            ir_index = build_time_index(ir_dir)
            for json_path in sorted(label_dir.glob("*.json")):
                source_time = parse_time(json_path)
                if source_time is None:
                    continue
                rgb_image = self._find_source_image(json_path, rgb_dir)
                nearest = nearest_by_time(ir_index, source_time)
                if rgb_image is None or nearest is None:
                    continue
                ir_image, diff = nearest
                if diff > self.max_time_diff:
                    continue
                samples.append({
                    "index": len(samples),
                    "json": json_path,
                    "rgb_image": rgb_image,
                    "ir_image": ir_image,
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

        with torch.no_grad():
            ir_image = read_image(str(sample["ir_image"]))
        ir_h, ir_w = ir_image.shape[-2:]

        boxes, ir_boxes, labels = [], [], []
        for shape in data.get("shapes", []):
            label = str(shape.get("label", ""))
            label = self.label_aliases.get(label, label)
            class_idx = self.categories.get(label)
            if class_idx is None or class_idx <= 0:
                continue
            bbox = shape_bbox(shape)
            if bbox is None:
                continue
            projected = project_bbox_by_homography(bbox, self.homography, ir_w, ir_h)
            if projected is None:
                continue
            boxes.append(bbox)
            ir_boxes.append(projected)
            labels.append(class_idx)

        if not boxes:
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            return {
                "boxes": empty_boxes,
                "ir_boxes": empty_boxes.clone(),
                "labels": torch.zeros((0,), dtype=torch.long),
            }

        return {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "ir_boxes": torch.as_tensor(ir_boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
        }


def initialize_lh_pairs(*args: Any, **kwargs: Any) -> LHPairsDataset:
    return LHPairsDataset.from_config(*args, **kwargs)
