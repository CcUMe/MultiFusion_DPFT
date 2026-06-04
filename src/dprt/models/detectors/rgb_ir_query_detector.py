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


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


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


def _normalize_bags_groups(groups, num_classes: int) -> List[List[int]]:
    if groups is None or groups == "auto":
        return [list(range(1, num_classes))]
    return [[int(class_idx) for class_idx in group] for group in groups]


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
        head_name: str = "linear_detection_head",
        bags_groups: List[List[int]] | None = None,
        num_reg_layers: int = 3,
        num_cls_layers: int = 3,
        head_bias: bool = False,
        head_dropout: float = 0.0,
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
        self.n_levels = len(feature_levels) if feature_levels is not None else None
        self.imagenet_normalize = imagenet_normalize
        self.use_bags = "bags" in head_name.lower() or bags_groups is not None

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
        attn_levels = self.n_levels if self.n_levels is not None else 4
        self.rgb_attention = MSDeformAttn(d_model=d_model, n_levels=attn_levels, n_heads=n_heads, n_points=n_points)
        self.ir_attention = MSDeformAttn(d_model=d_model, n_levels=attn_levels, n_heads=n_heads, n_points=n_points)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.head_dropout = head_dropout
        self.box_head = self._get_head_branch(4, num_reg_layers, head_bias, head_dropout)
        if self.use_bags:
            self.groups = _normalize_bags_groups(bags_groups, num_classes)
            self.group_sizes = [len(group) + 1 for group in self.groups]
            self.num_bags_logits = sum(self.group_sizes)
            self.objectness_head = self._get_head_branch(1, num_cls_layers, head_bias, head_dropout)
            self.bags_head = self._get_head_branch(self.num_bags_logits, num_cls_layers, head_bias, head_dropout)
        else:
            self.class_head = self._get_head_branch(num_classes, num_cls_layers, head_bias, head_dropout)
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
        bags_config = head_config.get("bags", {})
        return cls(
            inputs=model.get("inputs", ["camera_mono", "ir_image"]),
            backbones=backbones,
            necks=necks,
            d_model=model.get("d_model", head_config.get("in_channels", 256)),
            num_queries=model.get("num_queries", 400),
            num_classes=head_config.get("num_classes", model.get("num_classes", 8)),
            feature_levels=model.get("feature_levels", ["1", "2", "3", "4"]),
            n_heads=model.get("n_heads", 8),
            n_points=model.get("n_points", 4),
            dropout=model.get("dropout", 0.1),
            imagenet_normalize=model.get("imagenet_normalize", True),
            head_name=head_config.get("name", "linear_detection_head"),
            bags_groups=bags_config.get("groups", head_config.get("groups")),
            num_reg_layers=head_config.get("num_reg_layers", 3),
            num_cls_layers=head_config.get("num_cls_layers", 3),
            head_bias=head_config.get("bias", False),
            head_dropout=head_config.get("dropout", 0.0),
        )

    def _get_head_branch(self, out_channels: int, num_layers: int, bias: bool, dropout: float) -> nn.Module:
        branch = []
        for _ in range(num_layers - 1):
            branch.append(nn.Linear(self.d_model, self.d_model, bias=bias))
            branch.append(nn.ReLU())
            branch.append(nn.Dropout(dropout))
        branch.append(nn.Linear(self.d_model, out_channels, bias=bias))
        return nn.Sequential(*branch)

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
            selected_keys = list(features.keys())
        else:
            selected_keys = [level for level in self.feature_levels if level in features]
        expected_levels = self.n_levels if self.n_levels is not None else len(features)
        if len(selected_keys) != expected_levels:
            raise ValueError(f"Expected {expected_levels} FPN levels for {key}, got {selected_keys}.")
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

    def _bags_to_class_scores(self, objectness: torch.Tensor, bags_logits: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(
            bags_logits.shape[:-1] + (self.num_classes,),
            dtype=bags_logits.dtype,
            device=bags_logits.device,
        )
        object_prob = torch.sigmoid(objectness).squeeze(-1)
        scores[..., 0] = 1.0 - object_prob

        start = 0
        for group, size in zip(self.groups, self.group_sizes):
            logits = bags_logits[..., start:start + size]
            probs = F.softmax(logits, dim=-1)
            for local_idx, class_idx in enumerate(group):
                if 0 <= class_idx < self.num_classes:
                    scores[..., class_idx] = object_prob * probs[..., local_idx]
            start += size

        return scores

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

        box_raw = self.box_head(fused)
        centers = torch.sigmoid(box_raw[..., :2] + inverse_sigmoid(refs))
        sizes = torch.sigmoid(box_raw[..., 2:])
        boxes = torch.cat((centers, sizes), dim=-1)
        boxes_xyxy = torch.clamp(cxcywh_to_xyxy(boxes), 0.0, 1.0)
        out = {
            "boxes": boxes,
            "boxes_xyxy": boxes_xyxy,
            "reference_points": refs,
            "ir_reference_points": ir_refs,
        }
        if self.use_bags:
            objectness = self.objectness_head(fused)
            bags_logits = self.bags_head(fused)
            out.update({
                "objectness": objectness,
                "bags_logits": bags_logits,
                "class": self._bags_to_class_scores(objectness, bags_logits),
            })
        else:
            class_logits = self.class_head(fused)
            out.update({
                "class_logits": class_logits,
                "class": F.softmax(class_logits, dim=-1),
            })
        return out


def build_rgb_ir_query_detector(*args, **kwargs):
    return RGBIRQueryDetector.from_config(*args, **kwargs)
