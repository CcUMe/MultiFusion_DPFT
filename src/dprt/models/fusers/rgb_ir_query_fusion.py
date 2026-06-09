from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch

from torch import nn

from dprt.models.fusers.basempfusion import WeightedFusion
from dprt.models.layers import MSDeformAttn


class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int) -> None:
        super().__init__()

        layers = []
        for index in range(num_layers):
            src = in_channels if index == 0 else hidden_channels
            dst = out_channels if index == num_layers - 1 else hidden_channels
            layers.append(nn.Linear(src, dst))
            if index < num_layers - 1:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.layers(batch)


class RGBIRFusionLayer(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 n_levels: Optional[List[int]] = None,
                 n_heads: Optional[List[int]] = None,
                 n_points: Optional[List[int]] = None,
                 dropout: float = 0.0,
                 norm: bool = False,
                 reduction: str = 'weighted',
                 **kwargs) -> None:
        super().__init__()

        del kwargs

        self.d_model = d_model
        self.n_levels = n_levels if n_levels is not None else [4, 4]
        self.n_heads = n_heads if n_heads is not None else [1, 1]
        self.n_points = n_points if n_points is not None else [1, 1]
        self.dropout = dropout
        self.norm = norm
        self.reduction = reduction
        self.m_views = 2

        self.ms_deform_attn = nn.ModuleDict({
            'camera_mono': MSDeformAttn(
                d_model=self.d_model,
                n_levels=int(self.n_levels[0]),
                n_heads=int(self.n_heads[0]),
                n_points=int(self.n_points[0]),
            ),
            'ir_image': MSDeformAttn(
                d_model=self.d_model,
                n_levels=int(self.n_levels[1]),
                n_heads=int(self.n_heads[1]),
                n_points=int(self.n_points[1]),
            )
        })

        fusion_type = {
            'weighted': 'gated',
            'concat_weighted': 'concat_mlp',
            'attn_weighted': 'attention',
        }.get(self.reduction, 'gated')
        self.reduction_layer = WeightedFusion(self.d_model, self.m_views, fusion_type=fusion_type)
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    @staticmethod
    def _flatten_features(batch: Dict[str, torch.Tensor]):
        input_spatial_shapes = torch.stack(
            tuple(torch.as_tensor(level.shape[1:3], device=level.device) for level in batch.values()),
            dim=0
        )
        input_spatial_shapes = torch.atleast_2d(input_spatial_shapes)
        input_flatten = torch.cat(tuple(level.flatten(start_dim=1, end_dim=2) for level in batch.values()), dim=1)
        input_level_start_index = torch.cumsum(
            torch.as_tensor(
                [0] + [level.shape[1] * level.shape[2] for level in batch.values()],
                device=input_flatten.device,
            ),
            dim=0
        )[:-1]
        return input_flatten, input_spatial_shapes, input_level_start_index

    def forward_cross_attn(self,
                           query: torch.Tensor,
                           batch: Dict[str, torch.Tensor],
                           reference_points: torch.Tensor,
                           input_name: str,
                           query_positions: torch.Tensor = None) -> torch.Tensor:
        input_flatten, input_spatial_shapes, input_level_start_index = self._flatten_features(batch)
        ref_points = reference_points.unsqueeze(2).repeat(1, 1, len(batch), 1)

        out = self.ms_deform_attn[input_name](
            self.with_pos_embed(query, query_positions),
            ref_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
        )

        out = query + self.dropout1(out)
        if self.norm:
            out = self.norm1(out)

        return out

    def reduce(self,
               query: torch.Tensor,
               queries: torch.Tensor) -> torch.Tensor:
        return self.reduction_layer(query, queries)

    def forward(self,
                query: torch.Tensor,
                batch: List[Dict[str, torch.Tensor]],
                reference_points: List[torch.Tensor],
                query_positions: torch.Tensor = None) -> torch.Tensor:
        queries = torch.zeros(query.shape + (self.m_views,), dtype=query.dtype, device=query.device)

        queries[..., 0] = self.forward_cross_attn(
            query,
            batch[0],
            reference_points[0],
            'camera_mono',
            query_positions,
        )
        queries[..., 1] = self.forward_cross_attn(
            query,
            batch[1],
            reference_points[1],
            'ir_image',
            query_positions,
        )

        return self.reduce(query, queries)


class RGBIRQueryFusion(nn.Module):
    def __init__(self,
                 i_iter: int = 1,
                 m_views: int = 2,
                 d_model: int = 256,
                 d_ffn: int = 1024,
                 n_queries: int = 400,
                 n_levels: Optional[List[int]] = None,
                 n_heads: Optional[List[int]] = None,
                 n_points: Optional[List[int]] = None,
                 q_init: str = 'normal_',
                 dropout: float = 0.0,
                 norm: bool = True,
                 reduction: str = 'weighted',
                 activation: str = 'ReLU',
                 head: nn.Module = None,
                 **kwargs) -> None:
        super().__init__()

        del d_ffn, activation, kwargs

        self.i_iter = i_iter
        self.m_views = m_views
        self.d_model = d_model
        self.n_queries = n_queries
        self.n_levels = n_levels if n_levels is not None else [4, 4]
        self.n_heads = n_heads if n_heads is not None else [1, 1]
        self.n_points = n_points if n_points is not None else [1, 1]
        self.dropout = dropout
        self.norm = norm
        self.reduction = reduction
        self.q_init = getattr(nn.init, q_init)

        if self.m_views != 2:
            raise ValueError(f'RGBIRQueryFusion expects 2 views, got {self.m_views}.')

        if head is None:
            head = nn.Identity()

        self.fusion_layers = nn.ModuleDict({
            'fusion' + str(i): RGBIRFusionLayer(
                d_model=self.d_model,
                n_levels=self.n_levels,
                n_heads=self.n_heads,
                n_points=self.n_points,
                dropout=self.dropout,
                norm=self.norm,
                reduction=self.reduction,
            )
            for i in range(self.i_iter)
        })
        self.heads = self._get_clones(head, self.i_iter)
        self.query_embedding = nn.Embedding(self.n_queries, self.d_model)
        self.reference_embedding = MLP(2, self.d_model, self.d_model, 2)

        query = torch.empty((self.n_queries, self.d_model))
        self.query = nn.Parameter(query)

        self.reset_parameters()

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'RGBIRQueryFusion':
        return cls(**config, **kwargs)

    @staticmethod
    def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
        return nn.ModuleList([deepcopy(module) for _ in range(n)])

    def reset_parameters(self) -> None:
        self.q_init(self.query)
        nn.init.normal_(self.query_embedding.weight, std=0.02)

    @staticmethod
    def _project_rgb_to_ir(reference_points: torch.Tensor,
                           homography: torch.Tensor,
                           rgb_original_size: torch.Tensor,
                           ir_original_size: torch.Tensor) -> torch.Tensor:
        batch_size = reference_points.shape[0]
        rgb_h = rgb_original_size[:, 0].view(batch_size, 1)
        rgb_w = rgb_original_size[:, 1].view(batch_size, 1)
        ir_h = ir_original_size[:, 0].view(batch_size, 1)
        ir_w = ir_original_size[:, 1].view(batch_size, 1)

        rgb_x = reference_points[..., 0] * rgb_w
        rgb_y = reference_points[..., 1] * rgb_h
        points = torch.stack((rgb_x, rgb_y, torch.ones_like(rgb_x)), dim=-1)
        projected = torch.bmm(points, homography.transpose(1, 2))

        denom = torch.clamp(projected[..., 2], min=1e-6)
        ir_x = projected[..., 0] / denom
        ir_y = projected[..., 1] / denom

        reference_points = torch.dstack((
            ir_x / torch.clamp(ir_w, min=1.0),
            ir_y / torch.clamp(ir_h, min=1.0),
        ))
        reference_points = torch.clip(reference_points, min=0.0, max=1.0)

        return reference_points

    def get_reference_points(self,
                             query: torch.Tensor,
                             projection: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        rgb_reference_points = query[..., :2]
        ir_reference_points = self._project_rgb_to_ir(
            rgb_reference_points,
            projection[1]['homography'].to(dtype=query.dtype),
            projection[0]['original_size'].to(dtype=query.dtype),
            projection[1]['original_size'].to(dtype=query.dtype),
        )
        return [rgb_reference_points, ir_reference_points]

    def forward(self,
                batch: List[Dict[str, torch.Tensor]],
                shape: List[torch.Tensor],
                projection: List[Dict[str, torch.Tensor]],
                out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        del shape

        batch_size = out['center'].shape[0]
        query = self.query.unsqueeze(0).repeat(batch_size, 1, 1)
        query_pos = self.query_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_pos = query_pos + self.reference_embedding(out['center'][..., :2])

        for layer, head in zip(self.fusion_layers.values(), self.heads):
            reference_points = self.get_reference_points(out['center'][..., :2], projection)
            query = layer(query, batch, reference_points, query_pos)
            self.last_query = query
            out = head(query, out)
            out['ir_reference_points'] = reference_points[1]

        return out


def build_rgb_ir_query_fusion(*args, **kwargs):
    return RGBIRQueryFusion.from_config(*args, **kwargs)
