from __future__ import annotations

import math

from collections import OrderedDict
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from torch import nn

from dprt.models.backbones import build_backbone
from dprt.models.layers.ms_deform_attn import MSDeformAttn
from dprt.models.necks import build_neck


def _build_module(build_fn, module_config: Dict[str, Any], computing: Dict[str, Any]) -> nn.Module:
    return build_fn(module_config["name"], dict(computing | module_config))


def _make_reference_grid(num_queries: int) -> torch.Tensor:
    cols = int(math.ceil(math.sqrt(num_queries)))
    rows = int(math.ceil(num_queries / cols))
    ys = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
    xs = (torch.arange(cols, dtype=torch.float32) + 0.5) / cols
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    refs = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    return refs[:num_queries]


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    half_w = w * 0.5
    half_h = h * 0.5
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


class MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            src = in_channels if i == 0 else hidden_channels
            dst = out_channels if i == num_layers - 1 else hidden_channels
            layers.append(nn.Linear(src, dst))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RGBIRQueryDetector(nn.Module):
    def __init__(
        self,
        inputs: List[str],
        backbones: Dict[str, nn.Module],
        necks: Dict[str, nn.Module],
        d_model: int = 256,
        num_queries: int = 400,
        num_classes: int = 8,
        feature_levels: List[str] | None = None,
        n_heads: int = 8,
        n_points: int = 4,
        dropout: float = 0.1,
        imagenet_normalize: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.inputs = inputs
        self.rgb_key = "camera_mono"
        self.ir_key = "ir_image"
        self.backbones = nn.ModuleDict(backbones)
        self.necks = nn.ModuleDict(necks)
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.feature_levels = feature_levels
        self.imagenet_normalize = imagenet_normalize

        self.register_buffer("reference_points", _make_reference_grid(num_queries), persistent=False)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 1, 3),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 1, 3),
            persistent=False,
        )

        self.query_embed = nn.Parameter(torch.empty(num_queries, d_model))
        self.reference_embed = MLP(2, d_model, d_model, 2)
        self.rgb_attention = MSDeformAttn(d_model=d_model, n_levels=3, n_heads=n_heads, n_points=n_points)
        self.ir_attention = MSDeformAttn(d_model=d_model, n_levels=3, n_heads=n_heads, n_points=n_points)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.class_head = MLP(d_model, d_model, num_classes, 3)
        self.box_head = MLP(d_model, d_model, 4, 3)
        self._reset_parameters()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RGBIRQueryDetector":
        computing = config["computing"]
        model = config["model"]
        backbones = {
            key: _build_module(build_backbone, module_config, computing)
            for key, module_config in model.get("backbones", {}).items()
        }
        necks = {
            key: _build_module(build_neck, module_config, computing)
            for key, module_config in model.get("necks", {}).items()
        }
        head_config = model.get("head", {})
        return cls(
            inputs=model.get("inputs", ["camera_mono", "ir_image"]),
            backbones=backbones,
            necks=necks,
            d_model=model.get("d_model", head_config.get("in_channels", 256)),
            num_queries=model.get("num_queries", 400),
            num_classes=head_config.get("num_classes", model.get("num_classes", 8)),
            feature_levels=model.get("feature_levels", ["2", "3", "4"]),
            n_heads=model.get("n_heads", 8),
            n_points=model.get("n_points", 4),
            dropout=model.get("dropout", 0.1),
            imagenet_normalize=model.get("imagenet_normalize", True),
        )

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.query_embed, std=0.02)

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.imagenet_normalize:
            return image
        return (image - self.image_mean.to(image.device)) / self.image_std.to(image.device)

    def _extract_features(self, key: str, image: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        image = self._normalize_image(image)
        features = self.backbones[key](image)
        features = self.necks[key](features)
        if self.feature_levels is None:
            selected_keys = list(features.keys())[-3:]
        else:
            selected_keys = [level for level in self.feature_levels if level in features]
        if len(selected_keys) != 3:
            raise ValueError(f"Expected 3 FPN levels for {key}, got {selected_keys}.")
        return OrderedDict((level, features[level]) for level in selected_keys)

    @staticmethod
    def _flatten_features(features: OrderedDict[str, torch.Tensor]):
        flattened = []
        spatial_shapes = []
        for feat in features.values():
            b, h, w, c = feat.shape
            flattened.append(feat.reshape(b, h * w, c))
            spatial_shapes.append((h, w))
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=flattened[0].device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))
        return torch.cat(flattened, dim=1), spatial_shapes, level_start_index

    @staticmethod
    def _project_rgb_refs_to_ir(
        refs: torch.Tensor,
        homography: torch.Tensor,
        rgb_original_size: torch.Tensor,
        ir_original_size: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = refs.shape
        rgb_h = rgb_original_size[:, 0].view(b, 1)
        rgb_w = rgb_original_size[:, 1].view(b, 1)
        ir_h = ir_original_size[:, 0].view(b, 1)
        ir_w = ir_original_size[:, 1].view(b, 1)

        rgb_x = refs[..., 0] * rgb_w
        rgb_y = refs[..., 1] * rgb_h
        ones = torch.ones_like(rgb_x)
        pts = torch.stack((rgb_x, rgb_y, ones), dim=-1)
        projected = torch.bmm(pts, homography.transpose(1, 2))
        denom = torch.clamp(projected[..., 2], min=1e-6)
        ir_x = projected[..., 0] / denom
        ir_y = projected[..., 1] / denom
        ir_refs = torch.stack((ir_x / torch.clamp(ir_w, min=1.0), ir_y / torch.clamp(ir_h, min=1.0)), dim=-1)
        return torch.clamp(ir_refs, 0.0, 1.0)

    def forward(self, batch: Dict[str, torch.Tensor], *args, **kwargs) -> Dict[str, torch.Tensor]:
        rgb_features = self._extract_features(self.rgb_key, batch[self.rgb_key])
        ir_features = self._extract_features(self.ir_key, batch[self.ir_key])
        rgb_flat, rgb_shapes, rgb_level_index = self._flatten_features(rgb_features)
        ir_flat, ir_shapes, ir_level_index = self._flatten_features(ir_features)

        b = batch[self.rgb_key].shape[0]
        refs = self.reference_points.to(batch[self.rgb_key].device).unsqueeze(0).expand(b, -1, -1)
        ir_refs = self._project_rgb_refs_to_ir(
            refs,
            batch["homography_rgb_to_ir"].to(dtype=refs.dtype),
            batch["rgb_original_size"].to(dtype=refs.dtype),
            batch["ir_original_size"].to(dtype=refs.dtype),
        )

        query = self.query_embed.unsqueeze(0).expand(b, -1, -1) + self.reference_embed(refs)
        rgb_reference = refs.unsqueeze(2).expand(-1, -1, rgb_shapes.shape[0], -1)
        ir_reference = ir_refs.unsqueeze(2).expand(-1, -1, ir_shapes.shape[0], -1)

        rgb_query = self.rgb_attention(query, rgb_reference, rgb_flat, rgb_shapes, rgb_level_index)
        ir_query = self.ir_attention(query, ir_reference, ir_flat, ir_shapes, ir_level_index)
        fused = self.norm(query + self.fusion(torch.cat((rgb_query, ir_query), dim=-1)))

        class_logits = self.class_head(fused)
        boxes = torch.sigmoid(self.box_head(fused))
        boxes_xyxy = torch.clamp(cxcywh_to_xyxy(boxes), 0.0, 1.0)
        return {
            "class_logits": class_logits,
            "boxes": boxes,
            "boxes_xyxy": boxes_xyxy,
            "class": F.softmax(class_logits, dim=-1),
            "reference_points": refs,
            "ir_reference_points": ir_refs,
        }


def build_rgb_ir_query_detector(*args, **kwargs):
    return RGBIRQueryDetector.from_config(*args, **kwargs)
