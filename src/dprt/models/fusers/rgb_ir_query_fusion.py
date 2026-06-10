from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch

from torch import nn

from dprt.models import layers
from dprt.models.fusers.basempfusion import WeightedFusion
from dprt.models.layers import MSDeformAttn


def _select_modal_values(values, inputs: List[str], default: int) -> List[int]:
    if values is None:
        return [default] * len(inputs)
    if not isinstance(values, (list, tuple)):
        return [int(values)] * len(inputs)
    if len(values) == len(inputs):
        return [int(value) for value in values]
    if len(values) == 2:
        canonical = {'camera_mono': 0, 'ir_image': 1}
        return [int(values[canonical[input_name]]) for input_name in inputs]
    raise ValueError(f'Cannot align modal values {values} with inputs {inputs}.')


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
                 inputs: Optional[List[str]] = None,
                 d_model: int = 256,
                 d_ffn: int = 1024,
                 n_levels: Optional[List[int]] = None,
                 n_heads: Optional[List[int]] = None,
                 n_points: Optional[List[int]] = None,
                 ffn_layer: str = 'Linear',
                 activation: str = 'ReLU',
                 dropout: float = 0.0,
                 norm: bool = False,
                 reduction: str = 'weighted',
                 **kwargs) -> None:
        super().__init__()

        del kwargs

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.inputs = inputs if inputs is not None else ['camera_mono', 'ir_image']
        self.n_levels = _select_modal_values(n_levels, self.inputs, 4)
        self.n_heads = _select_modal_values(n_heads, self.inputs, 1)
        self.n_points = _select_modal_values(n_points, self.inputs, 1)
        self.ffn_layer = ffn_layer
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.reduction = reduction
        self.m_views = len(self.inputs)

        self.self_attn = nn.ModuleDict({
            input_name: nn.MultiheadAttention(
                self.d_model, int(self.n_heads[index]), dropout=self.dropout, batch_first=True
            )
            for index, input_name in enumerate(self.inputs)
        })
        self.ms_deform_attn = nn.ModuleDict({
            input_name: MSDeformAttn(
                d_model=self.d_model,
                n_levels=int(self.n_levels[index]),
                n_heads=int(self.n_heads[index]),
                n_points=int(self.n_points[index]),
            )
            for index, input_name in enumerate(self.inputs)
        })
        self.ffn1 = nn.ModuleDict({
            input_name: self._get_ffn_layer(self.ffn_layer, self.d_model, self.d_ffn)
            for input_name in self.inputs
        })
        self.activations = nn.ModuleDict({
            input_name: self._get_activation_fn(self.activation) for input_name in self.inputs
        })
        self.ffn2 = nn.ModuleDict({
            input_name: self._get_ffn_layer(self.ffn_layer, self.d_ffn, self.d_model)
            for input_name in self.inputs
        })

        fusion_type = {
            'weighted': 'gated',
            'concat_weighted': 'concat_mlp',
            'attn_weighted': 'attention',
        }.get(self.reduction, 'gated')
        self.reduction_layer = WeightedFusion(self.d_model, self.m_views, fusion_type=fusion_type)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)
        self.dropout4 = nn.Dropout(self.dropout)
        self.norm1 = nn.ModuleDict({name: nn.LayerNorm(self.d_model) for name in self.inputs})
        self.norm2 = nn.ModuleDict({name: nn.LayerNorm(self.d_model) for name in self.inputs})
        self.norm3 = nn.ModuleDict({name: nn.LayerNorm(self.d_model) for name in self.inputs})

    @staticmethod
    def _get_activation_fn(name: str) -> nn.Module:
        return getattr(nn, name)()

    @staticmethod
    def _get_ffn_layer(name: str, *args, **kwargs) -> nn.Module:
        try:
            return getattr(layers, name)(*args, **kwargs)
        except AttributeError:
            return getattr(nn, name)(*args, **kwargs)

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

    def forward_self_attn(self,
                          query: torch.Tensor,
                          input_name: str,
                          query_positions: torch.Tensor = None) -> torch.Tensor:
        q = k = self.with_pos_embed(query, query_positions)
        out = self.self_attn[input_name](query=q, key=k, value=query, need_weights=False)[0]
        out = query + self.dropout1(out)
        if self.norm:
            out = self.norm1[input_name](out)
        return out

    def forward_cross_attn(self,
                           query: torch.Tensor,
                           batch: Dict[str, torch.Tensor],
                           reference_points: torch.Tensor,
                           input_name: str,
                           query_positions: torch.Tensor = None,
                           reference_valid: torch.Tensor = None) -> torch.Tensor:
        input_flatten, input_spatial_shapes, input_level_start_index = self._flatten_features(batch)
        ref_points = reference_points.unsqueeze(2).repeat(1, 1, len(batch), 1)

        out = self.ms_deform_attn[input_name](
            self.with_pos_embed(query, query_positions),
            ref_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
        )

        if reference_valid is not None:
            out = out * reference_valid.unsqueeze(-1).to(dtype=out.dtype)
        out = query + self.dropout2(out)
        if self.norm:
            out = self.norm2[input_name](out)
        return out

    def forward_ffn(self, query: torch.Tensor, input_name: str) -> torch.Tensor:
        out = self.ffn2[input_name](
            self.dropout3(self.activations[input_name](self.ffn1[input_name](query)))
        )
        out = query + self.dropout4(out)
        if self.norm:
            out = self.norm3[input_name](out)
        return out

    def reduce(self,
               query: torch.Tensor,
               queries: torch.Tensor,
               reference_valid: List[torch.Tensor]) -> torch.Tensor:
        if self.reduction == 'weighted':
            valid = torch.stack(reference_valid, dim=-1)
            gates = self.reduction_layer.gate_mlp[:-1](query)
            gates = gates.masked_fill(~valid, torch.finfo(gates.dtype).min)
            gates = torch.softmax(gates, dim=-1).unsqueeze(2)
            return torch.sum(queries * gates, dim=-1)

        valid = torch.stack(reference_valid, dim=-1).unsqueeze(2)
        queries = torch.where(valid, queries, query.unsqueeze(-1))
        return self.reduction_layer(query, queries)

    def forward(self,
                query: torch.Tensor,
                batch: List[Dict[str, torch.Tensor]],
                reference_points: List[torch.Tensor],
                reference_valid: List[torch.Tensor],
                query_positions: torch.Tensor = None) -> torch.Tensor:
        queries = torch.zeros(query.shape + (self.m_views,), dtype=query.dtype, device=query.device)

        for index, input_name in enumerate(self.inputs):
            modal_query = self.forward_self_attn(query, input_name, query_positions)
            modal_query = self.forward_cross_attn(
                modal_query, batch[index], reference_points[index], input_name,
                query_positions, reference_valid[index]
            )
            queries[..., index] = self.forward_ffn(modal_query, input_name)

        if self.m_views == 1:
            return queries[..., 0]
        return self.reduce(query, queries, reference_valid)


class RGBIRQueryFusion(nn.Module):
    def __init__(self,
                 inputs: Optional[List[str]] = None,
                 i_iter: int = 1,
                 m_views: int = 2,
                 d_model: int = 256,
                 d_ffn: int = 1024,
                 n_queries: int = 400,
                 n_levels: Optional[List[int]] = None,
                 n_heads: Optional[List[int]] = None,
                 n_points: Optional[List[int]] = None,
                 q_init: str = 'normal_',
                 ffn_layer: str = 'Linear',
                 dropout: float = 0.0,
                 norm: bool = True,
                 reduction: str = 'weighted',
                 activation: str = 'ReLU',
                 head: nn.Module = None,
                 **kwargs) -> None:
        super().__init__()

        del kwargs

        self.i_iter = i_iter
        self.inputs = inputs if inputs is not None else ['camera_mono', 'ir_image']
        self.m_views = len(self.inputs)
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_queries = n_queries
        self.n_levels = _select_modal_values(n_levels, self.inputs, 4)
        self.n_heads = _select_modal_values(n_heads, self.inputs, 1)
        self.n_points = _select_modal_values(n_points, self.inputs, 1)
        self.ffn_layer = ffn_layer
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.reduction = reduction
        self.q_init = getattr(nn.init, q_init)

        invalid_inputs = set(self.inputs) - {'camera_mono', 'ir_image'}
        if invalid_inputs or not self.inputs:
            raise ValueError(f'RGBIRQueryFusion received unsupported inputs: {self.inputs}.')
        if len(self.n_levels) != self.m_views or len(self.n_heads) != self.m_views or len(self.n_points) != self.m_views:
            raise ValueError(
                'RGBIRQueryFusion expects one n_levels/n_heads/n_points value per input. '
                f'Got inputs={self.inputs}, n_levels={self.n_levels}, n_heads={self.n_heads}, n_points={self.n_points}.'
            )

        if head is None:
            head = nn.Identity()

        self.fusion_layers = nn.ModuleDict({
            'fusion' + str(i): RGBIRFusionLayer(
                inputs=self.inputs,
                d_model=self.d_model,
                d_ffn=self.d_ffn,
                n_levels=self.n_levels,
                n_heads=self.n_heads,
                n_points=self.n_points,
                ffn_layer=self.ffn_layer,
                activation=self.activation,
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
                           ir_original_size: torch.Tensor):
        batch_size = reference_points.shape[0]
        rgb_h = rgb_original_size[:, 0].view(batch_size, 1)
        rgb_w = rgb_original_size[:, 1].view(batch_size, 1)
        ir_h = ir_original_size[:, 0].view(batch_size, 1)
        ir_w = ir_original_size[:, 1].view(batch_size, 1)

        rgb_x = reference_points[..., 0] * rgb_w
        rgb_y = reference_points[..., 1] * rgb_h
        points = torch.stack((rgb_x, rgb_y, torch.ones_like(rgb_x)), dim=-1)
        projected = torch.bmm(points, homography.transpose(1, 2))

        denom = projected[..., 2]
        valid_denom = torch.isfinite(denom) & (denom.abs() > 1e-6)
        safe_denom = torch.where(valid_denom, denom, torch.ones_like(denom))
        ir_x = projected[..., 0] / safe_denom
        ir_y = projected[..., 1] / safe_denom

        reference_points = torch.dstack((
            ir_x / torch.clamp(ir_w, min=1.0),
            ir_y / torch.clamp(ir_h, min=1.0),
        ))
        reference_valid = (
            valid_denom
            & torch.isfinite(reference_points).all(dim=-1)
            & (reference_points[..., 0] >= 0.0)
            & (reference_points[..., 0] <= 1.0)
            & (reference_points[..., 1] >= 0.0)
            & (reference_points[..., 1] <= 1.0)
        )
        reference_points = torch.clip(reference_points, min=0.0, max=1.0)
        return reference_points, reference_valid

    def get_reference_points(self,
                             query: torch.Tensor,
                             projection: List[Dict[str, torch.Tensor]]):
        rgb_reference_points = query[..., :2]
        reference_points = []
        reference_valid = []
        for input_name, input_projection in zip(self.inputs, projection):
            if input_name == 'camera_mono':
                reference_points.append(rgb_reference_points)
                reference_valid.append(torch.ones(
                    rgb_reference_points.shape[:2], dtype=torch.bool, device=rgb_reference_points.device
                ))
            else:
                ir_reference_points, ir_reference_valid = self._project_rgb_to_ir(
                    rgb_reference_points,
                    input_projection['homography'].to(dtype=query.dtype),
                    input_projection['rgb_original_size'].to(dtype=query.dtype),
                    input_projection['original_size'].to(dtype=query.dtype),
                )
                reference_points.append(ir_reference_points)
                reference_valid.append(ir_reference_valid)
        return reference_points, reference_valid

    def forward(self,
                batch: List[Dict[str, torch.Tensor]],
                shape: List[torch.Tensor],
                projection: List[Dict[str, torch.Tensor]],
                out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        del shape

        batch_size = out['center'].shape[0]
        query = self.query.unsqueeze(0).repeat(batch_size, 1, 1)
        base_query_pos = self.query_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        for layer, head in zip(self.fusion_layers.values(), self.heads):
            query_pos = base_query_pos + self.reference_embedding(out['center'][..., :2])
            reference_points, reference_valid = self.get_reference_points(out['center'][..., :2], projection)
            query = layer(query, batch, reference_points, reference_valid, query_pos)
            self.last_query = query
            out = head(query, out)
            if 'ir_image' in self.inputs:
                ir_index = self.inputs.index('ir_image')
                out['ir_reference_points'] = reference_points[ir_index]
                out['ir_reference_valid'] = reference_valid[ir_index]

        return out


def build_rgb_ir_query_fusion(*args, **kwargs):
    return RGBIRQueryFusion.from_config(*args, **kwargs)
