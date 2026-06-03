from __future__ import annotations  # noqa: F407

from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from torch import nn

from dprt.models import layers
from dprt.models.layers import MSDeformAttn
from dprt.models.utils.transformations import cart2spher


# 在 mpfusion.py 文件开头（import 之后）添加：

class WeightedFusion(nn.Module):
    """加权融合模块，替换最大池化操作"""

    def __init__(self, d_model: int, m_views: int, fusion_type: str = 'gated'):
        """
        Args:
            d_model: 特征维度
            m_views: 视角数量
            fusion_type: 融合类型，可选 'gated', 'concat_mlp', 'attention'
        """
        super().__init__()
        self.d_model = d_model
        self.m_views = m_views
        self.fusion_type = fusion_type

        if fusion_type == 'gated':
            # 门控加权：为每个视角学习一个标量权重
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, m_views),
                nn.Softmax(dim=-1)
            )
        elif fusion_type == 'concat_mlp':
            # 拼接后MLP压缩
            self.fusion_mlp = nn.Sequential(
                nn.Linear(d_model * m_views, d_model * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
            )
        elif fusion_type == 'attention':
            # 升级: 使用 self-attn 跨模态融合 + 模态位置编码
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=8, dropout=0.1, batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
            # Learnable 模态位置编码 (区分不同传感器)
            self.modality_pos = nn.Parameter(torch.randn(m_views, d_model) * 0.02)
            # MODIFIED: 新增'sam_cross'类型，实现SAMFusion交叉注意力+池化
            # MODIFIED: 'sam_cross'类型，实现SAMFusion交叉注意力+池化（简化版，无距离）
        elif fusion_type == 'sam_cross':
            # SAMFusion式: 成对交叉注意力 (相机-LiDAR双向, 相机-雷达单向) + 权重池化
            self.cross_cam_lidar = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1, batch_first=True)  # C -> L
            self.cross_lidar_cam = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1, batch_first=True)  # L -> C
            self.cross_cam_radar = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1, batch_first=True)  # C -> R
            self.norm = nn.LayerNorm(d_model)
            self.modality_pos = nn.Parameter(torch.randn(m_views, d_model) * 0.02)
            # 修复: 权重MLP输入m_views=4 (模态均值), 输出4 (softmax weights)
            self.weight_mlp = nn.Sequential(
                nn.Linear(m_views, m_views // 2),  # [B,N,4] -> [B,N,2]
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(m_views // 2, m_views),  # -> [B,N,4]
                nn.Softmax(dim=-1)
            )
            # 模态dropout，提升鲁棒性
            self.mod_dropout = nn.Dropout(0.1)


        elif fusion_type == 'enhanced_gated':

            # 新增：增强门控融合（借鉴MSGF思路）

            # 1. 跨模态融合指导网络（用Ff生成权重）

            self.cross_guide_net = nn.Sequential(

                nn.Linear(d_model * 2, d_model),  # 融合LiDAR+Radar特征

                nn.ReLU(),

                nn.Dropout(0.1),

                nn.Linear(d_model, d_model // 2),

                nn.ReLU(),

            )

            # 2. 位置敏感门控生成器（生成逐点权重而非全局标量）

            self.spatial_gate = nn.Sequential(

                nn.Linear(d_model // 2, m_views),

                nn.Sigmoid()  # 使用Sigmoid而非Softmax，允许多模态并行贡献

            )

            # 3. 模态重要性加权（Softmax归一化）

            self.modality_weight = nn.Sequential(

                nn.Linear(d_model // 2, m_views),

                nn.Softmax(dim=-1)

            )

            # 4. 位置编码增强（可选）

            self.query_pos_proj = nn.Linear(d_model, d_model // 2)

            # 5. 门控阈值（动态过滤低贡献模态）

            self.gate_threshold = 0.3  # 可调参数，低于此值的模态被抑制


        else:

            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, query: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: 原始查询 (B, N, d_model)
            queries: 多视角查询 (B, N, d_model, m_views)
        Returns:
            融合后的查询 (B, N, d_model)
        """
        B, N, d_model, m_views = queries.shape

        if self.fusion_type == 'gated':
            # 使用原始查询计算门控权重
            gates = self.gate_mlp(query)  # (B, N, m_views)
            gates = gates.unsqueeze(2)  # (B, N, 1, m_views)

            # 加权求和
            weighted_queries = torch.sum(queries * gates, dim=-1)  # (B, N, d_model)
            return weighted_queries

        elif self.fusion_type == 'concat_mlp':
            # 拼接所有视角特征
            concat_queries = queries.view(B, N, d_model * m_views)
            fused = self.fusion_mlp(concat_queries)
            return fused + query  # 残差连接


        elif self.fusion_type == 'attention':

            # 升级: 跨模态自注意力融合

            modal_feats = queries.view(B * N, m_views, self.d_model)  # B*N, mviews, dmodel

            # 添加模态位置编码

            modal_pos = self.modality_pos.unsqueeze(0).expand(B * N, -1, -1)  # B*N, mviews, dmodel

            modal_feats = modal_feats + modal_pos

            # 自注意力: 让模态间相互关注

            fused_seq, _ = self.cross_attn(

                query=modal_feats, key=modal_feats, value=modal_feats

            )  # B*N, mviews, dmodel

            # 聚合: 均值融合 (可替换为 max 或加权 mean)

            fused = fused_seq.mean(dim=1)  # B*N, dmodel -> 跨模态融合

            fused = fused.view(B, N, self.d_model)

            return self.norm(fused + query)  # 残差 + 规范
            # MODIFIED: SAMFusion式交叉注意力池化（无距离自适应）

        elif self.fusion_type == 'sam_cross':
            # 假设视图顺序: 0=相机, 3=LiDAR, 1/2=雷达 (合并为radar)
            f_cam = queries[..., 0]  # (B, N, d_model)
            f_lidar = queries[..., 3]
            f_radar = torch.mean(queries[..., 1:3], dim=-1)  # 合并雷达 (B, N, d_model)

            # 双向交叉: 相机-LiDAR (SAMFusion Camera-Adaptive & LiDAR-Adaptive)
            c_enh, w_cl = self.cross_cam_lidar(query=f_cam, key=f_lidar, value=f_lidar)
            c_enh = self.norm(c_enh + f_cam)  # 残差
            l_enh, w_lc = self.cross_lidar_cam(query=f_lidar, key=c_enh, value=c_enh)
            l_enh = self.norm(l_enh + f_lidar)

            # 单向交叉: 增强相机-雷达 (避免稀疏LiDAR主导雷达)
            r_enh, w_cr = self.cross_cam_radar(query=c_enh, key=f_radar, value=f_radar)
            r_enh = self.norm(r_enh + f_radar)

            # 加权池化: softmax权重求和 (取代max)
            all_feats = torch.stack([c_enh, l_enh, r_enh, f_radar], dim=-1)  # (B, N, d_model, 4)
            # 修复权重计算: 先mean over d_model得模态向量 [B,N,4]，然后MLP(4->4)
            feat_mean = torch.mean(all_feats, dim=-2)  # [B,N,4]
            weights = self.weight_mlp(feat_mean)  # [B,N,4] -> [B,N,4] softmax
            weights = weights.unsqueeze(-2)  # [B,N,1,4] for broadcast

            fused = torch.sum(all_feats * weights, dim=-1)  # (B, N, d_model)
            fused = self.mod_dropout(fused)

            return self.norm(fused + query)  # 最终残差


        elif self.fusion_type == 'enhanced_gated':
            # 新增：增强门控融合（借鉴MSGF）
            # 步骤1: 提取代表性模态特征（假设m_views=4: [cam, radar1, radar2, lidar]）
            if m_views >= 4:
                f_lidar = queries[..., 3]  # (B, N, d_model) - LiDAR特征
                f_radar = torch.mean(queries[..., 1:3], dim=-1)  # (B, N, d_model) - 雷达平均

            else:
                # 简化为前两个视角
                f_lidar = queries[..., 0]
                f_radar = queries[..., -1] if m_views > 1 else queries[..., 0]

            # 步骤2: 构建跨模态融合特征Ff（类似MSGF的Ff = concat(Fl, Fr)）
            f_fusion = torch.cat([f_lidar, f_radar], dim=-1)  # (B, N, d_model*2)

            # 步骤3: 用Ff指导生成门控中间表示（加入位置编码增强）
            query_pos_feat = self.query_pos_proj(query)  # (B, N, d_model//2)
            fusion_guide = self.cross_guide_net(f_fusion)  # (B, N, d_model//2)
            fusion_guide = fusion_guide + query_pos_feat  # 位置敏感增强

            # 步骤4: 生成空间敏感门控权重（Sigmoid，支持多模态并行）
            spatial_gates = self.spatial_gate(fusion_guide)  # (B, N, m_views)

            # 步骤5: 应用阈值过滤（动态抑制低贡献模态，类似MSGF的⊙操作）
            gate_mask = (spatial_gates > self.gate_threshold).float()  # 二值mask
            spatial_gates = spatial_gates * gate_mask  # 过滤低权重

            # 步骤6: 生成模态重要性权重（Softmax归一化）
            modality_weights = self.modality_weight(fusion_guide)  # (B, N, m_views)

            # 步骤7: 组合门控：spatial_gates控制逐点过滤，modality_weights控制全局平衡
            combined_gates = spatial_gates * modality_weights  # (B, N, m_views)

            # 归一化（避免全零）
            gate_sum = combined_gates.sum(dim=-1, keepdim=True) + 1e-8
            combined_gates = combined_gates / gate_sum

            # 步骤8: 应用门控加权融合
            combined_gates = combined_gates.unsqueeze(2)  # (B, N, 1, m_views)
            weighted_queries = torch.sum(queries * combined_gates, dim=-1)  # (B, N, d_model)

            # 步骤9: 残差连接
            return weighted_queries + query
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

# 用于单视图的多尺度特征查询，支持自注意力、跨模态可变形注意力和前馈网络，处理BEV多层特征（如不同分辨率）
# Self-Attention → Cross-Attention → FFN
# 作用：单视角的多尺度特征查询，类似 Decoder 的一层。
class MLFusion(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 d_ffn: int = 1024,
                 n_levels: int = 1,
                 n_heads: int = 1,
                 n_points: int = 1,
                 ffn_layer: str = 'Linear',
                 activation: str = 'ReLU',
                 dropout: float = 0.0,
                 norm: bool = False,
                 **kwargs):
        """Multi-Level Fusion Transformer.

        Arguments:
            d_model: Hidden feature (channel) dimension of the
                attention modules (self and cross attention).
            d_ffn: Hidden feature (channel) dimension of the
                feed forward module.
            n_levels: Number of feature levels of the cross attention module.
            n_heads: Number of attention heads of the attention modules
                (self and cross attention).
            n_points: Number of sampling points per attention head
                (self and cross attention) and per feature level (cross attention).
        """
        # Initialize base class
        super().__init__()

        # Initialize instance attributes
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ffn_layer = ffn_layer
        self.activation = activation
        self.dropout = dropout
        self.norm = norm

        # Initialize self attention module
        self.self_attn = nn.MultiheadAttention(self.d_model, self.n_heads,
                                               dropout=self.dropout, batch_first=True)
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # Initialize deformable cross attention module
        self.flatten1 = nn.Flatten(start_dim=1, end_dim=2)
        self.ms_deform_attn = MSDeformAttn(self.d_model, self.n_levels,
                                           self.n_heads, self.n_points)
        self.dropout2 = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.d_model)

        # Initialize feed forward (ffn) module
        self.ffn1 = self._get_ffn_layer(self.ffn_layer, self.d_model, self.d_ffn)
        self.activation1 = self._get_activation_fn(self.activation)
        self.dropout3 = nn.Dropout(self.dropout)
        self.ffn2 = self._get_ffn_layer(self.ffn_layer, self.d_ffn, self.d_model)
        self.dropout4 = nn.Dropout(self.dropout)
        self.norm3 = nn.LayerNorm(self.d_model)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MLFusion:  # noqa: F821
        return cls(**config)

    @staticmethod
    def _get_activation_fn(name: str, *args, **kwargs) -> nn.Module:
        """Returns an activation function instance.

        Arguments:
            name: Name of the activation function module.

        Returns:
            Activation function module instance.
        """
        return getattr(nn, name)(*args, **kwargs)

    @staticmethod
    def _get_ffn_layer(name: str, *args, **kwargs) -> nn.Module:
        """Returns an feed forward network layer instance.

        Arguments:
            name: Name of the feed forward layer module.

        Retruns:
            Feed forward network module instance.
        """
        try:
            return getattr(layers, name)(*args, **kwargs)
        except AttributeError:
            return getattr(nn, name)(*args, **kwargs)
        except Exception as e:
            raise e

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """Returns a positional embedded tensor.

        Arguments:
            tensor: A tensor with shape (B, N, C)
            pos: A positional embedding with shape (B, N, C)

        Retruns:
            Positional embedded tensor.
        """
        return tensor if pos is None else tensor + pos

    def forward_self_attn(self,
                          query: torch.Tensor,
                          query_positions: torch.Tensor = None,
                          query_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns the self attended query features.

        Arguments:
            query: Query feature tensor with shape (B, N, d_model).
            query_positions: Positional embedding values of the query features
                with shape (B, N, d_model)

        Returns:
            out: Output feature tensor with shape (B, N, d_model).
        """
        # Apply positional embedding
        q = k = self.with_pos_embed(query, query_positions)

        # Apply self attention
        key_padding_mask = None if query_mask is None else ~query_mask
        out = self.self_attn(query=q, key=k, value=query,
                             key_padding_mask=key_padding_mask,
                             need_weights=False)[0]

        # Apply dropout
        out = query + self.dropout1(out)

        # Apply normalization
        if self.norm:
            out = self.norm1(out)

        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(out.dtype)

        return out

    def forward_cross_attn(self,
                           query: torch.Tensor,
                           batch: Dict[str, torch.Tensor],
                           reference_points: torch.Tensor,
                           query_positions: torch.Tensor = None,
                           query_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns query features based on the given batch feature levels and reference points.

        Arguments:
            query: A tensor of query featurs used during the attention with
                shape (B, N, d_model).
            batch: Dictionary of multi-level input features with length n_levels.
                The input batch represent the keys and values to attend to.
                The tensors are of shape (B, H, W, d_model)
            reference_points: A tensor representing normalized reference points
                with shape (B, N, 2).
            query_positions: Positional embedding values of the query features
                with shape (B, N, d_model)

        Returns:
            out: Fused multi-level output features with shape (B, N, d_model).
        """
        # Get input feature map dimensions
        input_spatial_shapes = torch.stack(
            tuple((torch.as_tensor(l.shape[1:3], device=l.device) for l in batch.values())),
            dim=0
        )
        input_spatial_shapes = torch.atleast_2d(input_spatial_shapes)

        # Flatten input features
        input_flatten = torch.cat(tuple((self.flatten1(l) for l in batch.values())), dim=1)

        # Determine flattend feature start indices
        input_level_start_index = torch.cumsum(
            torch.as_tensor(
                [0] + [l.shape[1] * l.shape[2] for l in batch.values()], device=query.device
                ),
            dim=0
        )[:-1]

        # Repeat reference points for each level
        ref_points = reference_points.unsqueeze(2).repeat(1, 1, len(batch), 1)

        # Query features from the current view
        out = self.ms_deform_attn(
                self.with_pos_embed(query, query_positions),
                ref_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index
        )

        # Apply dropout
        out = query + self.dropout2(out)

        # Apply normalization
        if self.norm:
            out = self.norm2(out)

        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(out.dtype)

        return out

    def forward_ffn(self, query: torch.Tensor) -> torch.Tensor:
        """Returns refined query features.

        Arguments:
            query: Query feature tensor with shape (B, N, d_model).

        Returns:
            out: Output feature tensor with shape (B, N, d_model).
        """
        # Apply feed forward layers
        out = self.ffn2(self.dropout3(self.activation1(self.ffn1(query))))

        # Apply dropout
        out = query + self.dropout4(out)

        # Apply normalization
        if self.norm:
            out = self.norm3(out)

        return out

    def forward(self,
                query: torch.Tensor,
                batch: Dict[str, torch.Tensor],
                reference_points: torch.Tensor,
                query_positions: torch.Tensor = None,
                query_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns query features based on the given input and reference points.

        Arguments:
            batch: Dictionary of multi-level input features with length n_levels.
                The input batch represent the keys and values to attend to.
                The tensors are of shape (B, H, W, d_model)
            query: A tensor of query featurs used during the attention with
                shape (B, N, d_model).
            reference_points: A tensor representing normalized reference points
                with shape (B, N, 2).
            query_positions: Positional embedding values of the query features
                with shape (B, N, d_model)

        Returns:
            out: Fused multi-level output features with shape (B, N, d_model).
        """
        # Self attention: Self attend to the queries
        out = self.forward_self_attn(query=query,
                                     query_positions=query_positions,
                                     query_mask=query_mask)

        # Cross attention: Cross attend to multi level features
        out = self.forward_cross_attn(query=out, batch=batch,
                                      reference_points=reference_points,
                                      query_positions=query_positions,
                                      query_mask=query_mask)

        # FFN: Propagate attended features
        out = self.forward_ffn(query=out)
        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(out.dtype)

        return out

# 作用：对多个视角独立查询，然后融合结果。
class MPFusion(nn.Module):
    def __init__(self,
                 m_views: int,
                 d_model: int = 256,
                 d_ffn: int = 1024,
                 n_levels: List[int] = None,
                 n_heads: List[int] = None,
                 n_points: List[int] = None,
                 ffn_layer: str = 'Linear',
                 activation: str = 'ReLU',
                 dropout: float = 0.0,
                 norm: bool = False,
                 reduction: str = 'mean',
                 **kwargs):
        """Multi-Perspective Fusion Transformer.

        Arguments:
            m_views: Number of perspective views to query from.
            d_model: Hidden feature (channel) dimension.
            n_levels: Number of feature levels for each view.
            n_heads: Number of attention heads for each view.
            n_points: Number of sampling points per attention head and
                per feature level for each view.
            reduction: Reduction mode to fuse the queries of multiple views.
                One of either mean, max, unary, linear, cross-attn or ffn.
        """
        # Initialize base class
        super().__init__()

        # Check input arguments
        if reduction not in {
            'mean', 'max', 'unary', 'linear', 'cross-attn', 'ffn',
            'weighted', 'concat_weighted', 'attn_weighted', 'sam_cross_pool','enhanced_gated'
        }:
            raise ValueError(
                f"The reduction mode must be one of either "
                f"'mean', 'max', 'unary', 'linear', 'cross-attn', 'ffn', "
                f"'weighted','sam_cross_pool',,'enhanced_gated','concat_weighted' or 'attn_weighted' but {reduction} "
                f"was given!"
            )

        # Initialize instance attributes
        self.m_views = m_views
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_levels = n_levels if n_levels is not None else [1] * m_views
        self.n_heads = n_heads if n_heads is not None else [1] * m_views
        self.n_points = n_points if n_points is not None else [1] * m_views
        self.ffn_layer = ffn_layer
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.reduction = reduction

        # Initialize module layers (one for each view)
        self.ml_fusion_layers = nn.ModuleDict({
            'ms_deform_attn' + str(v):
            MLFusion(self.d_model, self.d_ffn, l, h, p,
                     self.ffn_layer, self.activation, self.dropout, self.norm)
            for v, l, h, p in zip(range(self.m_views), self.n_levels, self.n_heads, self.n_points)
        })

        # Initialize reduction (fusion) layer
        self.reduction_layer = self._init_reduction(self.reduction)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MPFusion:  # noqa: F821
        return cls(**config)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """Returns a positional embedded tensor.

        Arguments:
            tensor: A tensor with shape (B, N, C)
            pos: A positional embedding with shape (B, N, C)

        Retruns:
            Positional embedded tensor.
        """
        return tensor if pos is None else tensor + pos

    @staticmethod
    def _get_activation_fn(name: str, *args, **kwargs) -> nn.Module:
        """Returns an activation function instance.

        Arguments:
            name: Name of the activation function module.

        Returns:
            Activation function module instance.
        """
        return getattr(nn, name)(*args, **kwargs)

    @staticmethod
    def _get_ffn_layer(name: str, *args, **kwargs) -> nn.Module:
        """Returns an feed forward network layer instance.

        Arguments:
            name: Name of the feed forward layer module.

        Retruns:
            Feed forward network module instance.
        """
        try:
            return getattr(layers, name)(*args, **kwargs)
        except AttributeError:
            return getattr(nn, name)(*args, **kwargs)
        except Exception as e:
            raise e

    def _init_reduction(self, reduction: str) -> Union[Callable, nn.Module]:
        """Returns a reduction module instance.

        Arguments:
            reduction: Selected reduction mode.

        Returns:
            Reduction module instance.
        """
        if reduction == 'mean':
            return partial(torch.mean, dim=-1)

        if reduction == 'max':
            return partial(torch.max, dim=-1)

        if reduction == 'unary':
            return layers.Unary1d(in_channels=self.m_views * self.d_model,
                                  out_channels=self.d_model,
                                  bias=False, channels_last=True)

        if reduction == 'linear':
            return nn.Linear(in_features=self.m_views * self.d_model, out_features=self.d_model,
                             bias=False)

        if reduction == 'cross-attn':
            return nn.MultiheadAttention(self.d_model, min(self.n_heads), dropout=self.dropout,
                                         kdim=self.d_model * self.m_views,
                                         vdim=self.d_model * self.m_views, batch_first=True)

        if reduction == 'ffn':
            return nn.ModuleDict({
                'ffn1': self._get_ffn_layer(self.ffn_layer, self.m_views * self.d_model,
                                            self.m_views * self.d_model),
                'activation1': self._get_activation_fn(self.activation),
                'dropout1': nn.Dropout(self.dropout),
                'ffn2': self._get_ffn_layer(self.ffn_layer, self.m_views * self.d_model,
                                            self.d_model),
                'downsample1': self._get_ffn_layer(self.ffn_layer, self.m_views * self.d_model,
                                                   self.d_model),
                'dropout2': nn.Dropout(self.dropout),
                'norm1': nn.LayerNorm(self.d_model),
            })
            # 新增加权融合选项
        if reduction == 'weighted':
            return WeightedFusion(self.d_model, self.m_views, fusion_type='gated')
        if reduction == 'concat_weighted':
            return WeightedFusion(self.d_model, self.m_views, fusion_type='concat_mlp')
        if reduction == 'attn_weighted':
            return WeightedFusion(self.d_model, self.m_views, fusion_type='attention')
        if reduction == 'sam_cross_pool':
            return WeightedFusion(self.d_model, self.m_views, fusion_type='sam_cross')
        if reduction == 'enhanced_gated':
            return WeightedFusion(self.d_model, self.m_views, fusion_type='enhanced_gated')
        raise ValueError(f"Unsupported reduction: {reduction}")

    def reduce(self,
               query: torch.Tensor,
               queries: torch.Tensor,
               query_positions: torch.Tensor) -> torch.Tensor:
        """Applies the selected reduction to the queries.

        Arguments:
            query: Original input query with shape (B, N, d_model)
            queries: Multi-view queries with shape (B, N, d_model, m_views)
            query_positions: Positional embedding values of the query features
                with shape (B, N, d_model)

        Returns:
            Reduced (fused) queries with shape (B, N, d_model)
        """
        if self.reduction in {'mean', 'max'}:
            return self.reduction_layer(queries)

        if self.reduction in {'unary', 'linear'}:
            # Get query dimensions
            B, N = query.shape[:2]

            return self.reduction_layer(queries.view(B, N, self.d_model * self.m_views))

        if self.reduction == 'cross-attn':
            # Get query dimensions
            B, N = query.shape[:2]

            return self.reduction_layer(
                query=self.with_pos_embed(query, query_positions),
                key=queries.view(B, N, self.d_model * self.m_views),
                value=queries.view(B, N, self.d_model * self.m_views),
                need_weights=False)[0]

        if self.reduction == 'ffn':
            # Get query dimensions
            B, N = query.shape[:2]

            queries = queries.view(B, N, self.d_model * self.m_views)

            # Apply ffn (similar to residual block)
            out = self.reduction_layer['ffn1'](queries)
            out = self.reduction_layer['activation1'](out)
            out = self.reduction_layer['dropout1'](out)
            out = self.reduction_layer['ffn2'](out)
            out = self.reduction_layer['dropout2'](out)

            queries = self.reduction_layer['downsample1'](queries)

            out = queries + out

            if self.norm:
                out = self.reduction_layer['norm1'](out)

            return out

        # 新增对加权融合的支持
        if self.reduction in {'weighted', 'concat_weighted', 'attn_weighted', 'sam_cross_pool','enhanced_gated'}:
            return self.reduction_layer(query, queries)

    def forward(self,
                query: torch.Tensor,
                batch: List[Dict[str, torch.Tensor]],
                reference_points: List[torch.Tensor],
                query_positions: torch.Tensor,
                query_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns fused query features based on the given input and reference points.

        Arguments:
            batch: List of ordered dictionaries mapping a level to a tensor.
                The list has length m_views and each dict has length n_levels.
                The input batch represent the keys and values to attend to.
                The tensors are of shape (B, H, W, d_model)
            query: A tensor of query featurs used during the attention with
                shape (B, N, d_model).
            reference_points: A list of tensors representing normalized
                reference points. The list has length m_views and the tensors
                have shape (B, N, 2).
            query_positions: Positional embedding values of the query features
                with shape (B, N, d_model)

        Returns:
            out: Fused output features with shape (B, N, d_model).
        """

        # print(
        #     f"Input: query.shape={query.shape}, len(batch)={len(batch)}, len(reference_points)={len(reference_points)}")
        # print(f"self.m_views={self.m_views}, ml_fusion_layers keys={list(self.ml_fusion_layers.keys())}")

        # Initialize queries with shape (query, m_views)
        queries = torch.zeros(query.shape + (self.m_views, ),
                              dtype=query.dtype, device=query.device)

        # print(f"Initialized queries.shape={queries.shape}")  # 期望 [4, ?, 16, 4]

        # print(f"batch length: {len(batch)}")  # 应为4
        # print(f"m_views: {self.m_views}")
        # print(f"queries shape after loop: {queries.shape}")  # 期望 (B, N, d_model, 4)

        # Define iterator
        iterator = zip(self.ml_fusion_layers.values(), batch, reference_points)
        # print(f"iterator len estimate: min({len(self.ml_fusion_layers)}, {len(batch)}, {len(reference_points)})")

        for i, (ml_fusion_layer, levels, ref_points) in enumerate(iterator):
            # print(
            #     f"Loop i={i}: levels.keys={list(levels.keys()) if isinstance(levels, dict) else 'not dict'}, ref_points.shape={ref_points.shape}")

            # Query features from the current view
            queries[..., i] = ml_fusion_layer(
                    query,
                    levels,
                    ref_points,
                    query_positions,
                    query_mask
            )
            # print(f"After i={i}: queries[..., {i}].shape={queries[..., i].shape}")

        # print(f"Final queries.shape={queries.shape}")
        # Fuse multi perspective query features
        out = self.reduce(query, queries, query_positions)
        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(out.dtype)

        return out

#和vlm维度对齐，新增投影层
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=16, out_dim=2560, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),    # 16 -> 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # 128 -> 128 (可选中间层)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),     # 128 -> 2560
            nn.LayerNorm(out_dim)  # 稳定输出，匹配VLM输入规范
        )

    def forward(self, x):
        return self.proj(x)


# 顶层类，支持i_iter次迭代融合，每个迭代用MPFusion+head精炼输出，适用于端到端检测
class IMPFusion(nn.Module):
    def __init__(self,
                 i_iter: int = 1,
                 m_views: int = 1,
                 d_model: int = 256,
                 d_ffn: int = 1024,
                 n_queries: int = 100,
                 n_levels: List[int] = None,
                 n_heads: List[int] = None,
                 n_points: List[int] = None,
                 q_init: str = 'uniform_',
                 ffn_layer: str = 'Linear',
                 activation: str = 'ReLU',
                 dropout: float = 0.0,
                 norm: bool = False,
                 reduction: str = 'mean',
                 head: nn.Module = None,
                 query_pruning: bool = False,
                 # stage_keep_ratios: Optional[List[float]] = None,
                 stage_keep_counts: Optional[List[int]] = None,
                 early_exit: bool = True,
                 early_exit_iter: int = 1,
                 exit_box_threshold: float = 7,
                 exit_score_threshold: float = 0.7,
                 exit_center_threshold: float = 0.2,
                 exit_size_threshold: float = 0.1,
                 exit_angle_threshold: float = 0.1,
                 score_key: str = 'class',
                 **kwargs):
        """
        Iterative Multi-Perspective Fusion Transformer

        这是整个多视角/多模态融合的顶层模块。
        它的主流程是：
            1. 初始化一组可学习 query
            2. 每一轮根据上一轮预测中心生成 reference points
            3. 调用 MPFusion 做多视角特征查询与融合
            4. 调用检测头 head 输出 center / class / size / angle 等
            5. 可选：做 query 分阶段裁剪（query_pruning）
            6. 可选：做 query early exit
            7. 剩余 query 进入下一轮 refinement

        新增的两个实时性优化：
            - query_pruning：每轮保留 top-K query 进入下一轮
            - early_exit：高置信且框稳定的 query 提前退出后续 refinement

        参数说明：
            i_iter: refinement 迭代次数
            m_views: 视角/模态数量
            d_model: query 特征维度
            d_ffn: FFN 隐层维度
            n_queries: 初始 query 数量
            n_levels: 每个视角的多尺度层数
            n_heads: 每个视角的注意力头数
            n_points: 每个头的采样点数
            q_init: query 参数初始化方式
            reduction: 多视角融合方式
            head: 每轮输出头，用于预测检测结果
            query_pruning: 是否启用分阶段裁剪 query
            stage_keep_ratios: 每轮保留比例，例如 [0.4, 0.2]
            stage_keep_counts: 每轮保留数量，例如 [160, 80]
            early_exit: 是否启用 query 提前退出
            early_exit_iter: 从第几轮开始允许退出（0-based）
            exit_score_threshold: early exit 的分类阈值
            exit_center_threshold: early exit 的中心稳定阈值
            exit_size_threshold: early exit 的尺寸稳定阈值
            exit_angle_threshold: early exit 的角度稳定阈值
            score_key: 从 out 中哪个字段取 query 分数，默认 'class'
        """
        super().__init__()

        # -----------------------------
        # 基础配置
        # -----------------------------
        self.i_iter = i_iter
        self.m_views = m_views
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_queries = n_queries
        self.n_levels = n_levels if n_levels is not None else [1] * m_views
        self.n_heads = n_heads if n_heads is not None else [1] * m_views
        self.n_points = n_points if n_points is not None else [1] * m_views
        self.ffn_layer = ffn_layer
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.reduction = reduction

        # -----------------------------
        # query 优化相关配置
        # -----------------------------
        self.query_pruning = query_pruning
        # self.stage_keep_ratios = stage_keep_ratios
        self.stage_keep_counts = stage_keep_counts
        self.early_exit = early_exit
        self.early_exit_iter = early_exit_iter
        self.exit_box_threshold = exit_box_threshold
        self.exit_score_threshold = exit_score_threshold
        self.exit_center_threshold = exit_center_threshold
        self.exit_size_threshold = exit_size_threshold
        self.exit_angle_threshold = exit_angle_threshold
        self.score_key = score_key if score_key is not None else 'class'

        # query 初始化函数，例如 nn.init.uniform_
        self.q_init = getattr(nn.init, q_init)

        if head is None:
            head = nn.Identity()

        # -----------------------------
        # 每一轮一个 MPFusion
        # fusion0, fusion1, fusion2, ...
        # -----------------------------
        self.mpfusion = nn.ModuleDict({
            'fusion' + str(i):
            MPFusion(self.m_views, self.d_model, self.d_ffn, self.n_levels,
                     self.n_heads, self.n_points, self.ffn_layer, self.activation,
                     self.dropout, self.norm, self.reduction)
            for i in range(self.i_iter)
        })

        # 每一轮一个检测头
        self.heads = self._get_clones(head, self.i_iter)

        # query 位置编码
        self.query_embedding = nn.Embedding(self.n_queries, self.d_model)

        # 可学习 query 内容向量
        query = torch.empty((self.n_queries, self.d_model))
        self.query = nn.Parameter(query)

        self.reset_parameters()

        # 额外投影头，和 VLM 维度对齐（你原来的设计）
        self.projection = ProjectionHead(self.d_model, 2560)

    def _ensure_runtime_attrs(self):
        if not hasattr(self, 'query_pruning'):
            self.query_pruning = False
        if not hasattr(self, 'stage_keep_counts'):
            # 从第二轮开始定义，省略第一轮
            self.stage_keep_counts = [380, 350, 300]
        if not hasattr(self, 'early_exit'):
            self.early_exit = True
        if not hasattr(self, 'early_exit_iter'):
            self.early_exit_iter = 1
        if not hasattr(self, 'exit_score_threshold'):
            self.exit_score_threshold = 0.5
        if not hasattr(self, 'exit_center_threshold'):
            self.exit_center_threshold = 0.2
        if not hasattr(self, 'exit_size_threshold'):
            self.exit_size_threshold = 0.1
        if not hasattr(self, 'exit_angle_threshold'):
            self.exit_angle_threshold = 0.1
        if not hasattr(self, 'score_key'):
            self.score_key = 'class'
        if not hasattr(self, 'exit_box_threshold'):
            self.exit_box_threshold = 1.5

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> IMPFusion:
        """从配置字典创建模块实例"""
        return cls(**config, **kwargs)

    @staticmethod
    def _get_clones(module: nn.Module, n: int) -> nn.Module:
        """
        复制 n 份子模块。
        通常用于：
            - 每轮一个 head
        """
        return nn.ModuleList([deepcopy(module) for i in range(n)])

    def reset_parameters(self) -> None:
        """初始化 query 参数"""
        self.q_init(self.query)

    @staticmethod
    def _batched_gather_tensor(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        按 batch 维度对 query 维（dim=1）做 gather。

        用途：
            当我们做 top-k query 裁剪时，需要把 query / query_pos / out['center']
            等所有“第 1 维是 query 维”的张量同步裁掉。

        输入：
            tensor: 形状通常为 (B, N, ...)
            index:  形状 (B, K)，表示每个 batch 保留哪些 query

        输出：
            gather 后形状通常为 (B, K, ...)
        """
        if tensor is None:
            return None
        if tensor.ndim < 2:
            return tensor

        index_expanded = index
        for _ in range(tensor.ndim - 2):
            index_expanded = index_expanded.unsqueeze(-1)
        index_expanded = index_expanded.expand(*index.shape, *tensor.shape[2:])
        return torch.gather(tensor, 1, index_expanded)

    def _batched_gather_out(self, out: Dict[str, torch.Tensor], index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        对输出字典 out 中所有与 query 对齐的字段做 batched gather。

        例如 out 里可能有：
            class:  (B, N, num_classes)
            center: (B, N, 3)
            size:   (B, N, 3)
            angle:  (B, N, 2)

        当 query 被 top-k 裁剪后，这些字段也必须同步裁剪。
        """
        gathered = {}
        for key, value in out.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == index.shape[0]:
                if value.shape[1] == 0:
                    gathered[key] = value
                elif value.shape[1] >= int(index.max().item()) + 1:
                    gathered[key] = self._batched_gather_tensor(value, index)
                else:
                    gathered[key] = value
            else:
                gathered[key] = value
        return gathered

    def _get_query_scores(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.score_key not in out:
            raise KeyError(
                f"Expected score key '{self.score_key}' in output dict, "
                f"but found {list(out.keys())}"
            )

        logits = out[self.score_key]

        # 二分类/多分类都先转成概率风格分数
        if logits.ndim == 2:
            return logits.sigmoid()

        return logits.sigmoid().max(dim=-1).values

    @staticmethod
    def _masked_topk(scores: torch.Tensor, valid_mask: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在 valid query 上做 top-k。

        为什么需要 valid_mask：
            因为在 query_pruning / early_exit 后，不同 batch 样本剩余 query 数可能不同。
            这里用 valid_mask 标记哪些 query 还有效，避免 padding query 被选中。

        返回：
            topk_idx:       (B, k)   每个 batch 选出的 query 索引
            new_valid_mask: (B, k)   top-k 后哪些位置是真实有效的
        """
        B, N = scores.shape
        k = max(1, min(k, N))

        # 无效 query 赋 -inf，确保不会进 top-k
        masked_scores = scores.masked_fill(~valid_mask, float('-inf'))

        topk_idx = torch.topk(masked_scores, k=k, dim=1).indices

        # 有些 batch 的有效 query 本来就少于 k，因此需要构造新的有效 mask
        valid_counts = valid_mask.sum(dim=1).clamp(max=k)
        range_k = torch.arange(k, device=scores.device).unsqueeze(0)
        new_valid_mask = range_k < valid_counts.unsqueeze(1)

        # 无效位置补成 0，避免 gather 越界
        topk_idx = torch.where(new_valid_mask, topk_idx, torch.zeros_like(topk_idx))
        return topk_idx, new_valid_mask

    @staticmethod
    def _pack_continue_mask(continue_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        把“继续参与下一轮 refinement 的 query”重新打包成紧凑表示。

        输入：
            continue_mask: (B, N)，True 表示该 query 继续下一轮

        返回：
            packed_indices:    (B, Kmax)
            packed_valid_mask: (B, Kmax)

        解释：
            不同 batch 中剩余 query 数可能不同。
            为了还能拼成一个 batch，需要 padding 到同一个长度 Kmax。
            packed_valid_mask 用来标识哪些位置是真 query，哪些是 padding。
        """
        B, N = continue_mask.shape
        counts = continue_mask.sum(dim=1)
        max_count = int(counts.max().item()) if counts.numel() > 0 else 0

        if max_count <= 0:
            device = continue_mask.device
            return (
                torch.zeros((B, 1), dtype=torch.long, device=device),
                torch.zeros((B, 1), dtype=torch.bool, device=device)
            )

        packed_indices = []
        packed_valid_mask = []
        base = torch.arange(max_count, device=continue_mask.device)

        for b in range(B):
            idx = torch.nonzero(continue_mask[b], as_tuple=False).squeeze(-1)
            num = idx.numel()
            valid = base < num

            if num == 0:
                padded = torch.zeros(max_count, dtype=torch.long, device=continue_mask.device)
            else:
                # padding 时重复第一个合法索引，仅用于占位
                pad_value = idx[0]
                if num < max_count:
                    pad = pad_value.repeat(max_count - num)
                    padded = torch.cat([idx, pad], dim=0)
                else:
                    padded = idx[:max_count]

            packed_indices.append(padded)
            packed_valid_mask.append(valid)

        return torch.stack(packed_indices, dim=0), torch.stack(packed_valid_mask, dim=0)

    def _compute_exit_mask(
            self,
            curr_out: Dict[str, torch.Tensor],
            prev_out: Dict[str, torch.Tensor],
            scores: torch.Tensor,
            valid_mask: torch.Tensor,
            iter: int
    ) -> torch.Tensor:
        if prev_out is None:
            return torch.zeros_like(valid_mask)

        eps = 1e-8
        min_size = getattr(self, "exit_min_size", 0.1)

        lambda_center = getattr(self, "exit_lambda_center", 1.0)
        lambda_size = getattr(self, "exit_lambda_size", 0.5)
        lambda_angle = getattr(self, "exit_lambda_angle", 0.15)

        use_score_gate = getattr(self, "exit_use_score_gate", False)
        score_threshold = getattr(self, "exit_score_threshold", 0.3)

        prev_center = prev_out["center"][..., :3]
        curr_center = curr_out["center"][..., :3]

        prev_size = prev_out["size"][..., :3].abs().clamp_min(eps)
        curr_size = curr_out["size"][..., :3].abs().clamp_min(eps)

        avg_size = 0.5 * (prev_size + curr_size)
        avg_size = avg_size.clamp_min(min_size)

        xy_scale = avg_size[..., :2].mean(dim=-1, keepdim=True).clamp_min(min_size)
        z_scale = avg_size[..., 2:3].clamp_min(min_size)

        center_diff_xy = (curr_center[..., :2] - prev_center[..., :2]).abs() / xy_scale
        center_diff_z = (curr_center[..., 2:3] - prev_center[..., 2:3]).abs() / z_scale
        center_delta_rel = torch.cat([center_diff_xy, center_diff_z], dim=-1).norm(dim=-1)

        size_delta_rel = ((curr_size - prev_size).abs() / avg_size).mean(dim=-1)

        prev_angle = prev_out.get("angle", None)
        curr_angle = curr_out.get("angle", None)
        if prev_angle is not None and curr_angle is not None:
            prev_angle_unit = prev_angle / prev_angle.norm(dim=-1, keepdim=True).clamp_min(eps)
            curr_angle_unit = curr_angle / curr_angle.norm(dim=-1, keepdim=True).clamp_min(eps)
            angle_delta = (curr_angle_unit - prev_angle_unit).norm(dim=-1)
        else:
            angle_delta = center_delta_rel.new_zeros(center_delta_rel.shape)

        box_delta = (
                lambda_center * center_delta_rel +
                lambda_size * size_delta_rel +
                lambda_angle * angle_delta
        )

        stable = box_delta <= self.exit_box_threshold

        return valid_mask & stable

    @staticmethod
    def _append_to_final_storage(storage: List[Dict[str, Any]],
                                 out: Dict[str, torch.Tensor],
                                 scores: torch.Tensor,
                                 valid_mask: torch.Tensor) -> None:
        """
        把当前轮已经“完成”的 query 存入最终结果缓存。

        这些“完成”的 query 包括：
            - 最后一轮剩下的所有有效 query
            - early exit 的 query

        为什么要单独缓存：
            因为 early exit 的 query 不会再参与后续 refinement，
            但它们仍然是最终检测结果的一部分。
        """
        B = scores.shape[0]
        for b in range(B):
            mask = valid_mask[b]
            if mask.sum().item() == 0:
                continue

            # Scores are only used for ordering; keep prediction fields below
            # attached to autograd so training losses can backpropagate.
            # 保存分数，后面用于排序
            storage[b]['scores'].append(scores[b][mask].detach())

            # 保存各字段
            for key, value in out.items():
                if not isinstance(value, torch.Tensor) or value.ndim < 2:
                    continue
                if value.shape[0] != B or value.shape[1] != valid_mask.shape[1]:
                    continue
                storage[b]['fields'].setdefault(key, [])
                storage[b]['fields'][key].append(value[b][mask])

    @staticmethod
    def _build_final_output(storage: List[Dict[str, Any]], template_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        把各轮缓存下来的 query 结果重新组装成最终输出。

        处理逻辑：
            1. 收集所有字段名
            2. 对每个 batch，把所有退出/保留的 query 拼接起来
            3. 按 score 从高到低排序
            4. padding 到 batch 内统一长度
            5. 生成 valid_mask，标记最终哪些位置是有效 query

        返回：
            final_out: dict，结构与 out 类似，并额外带 valid_mask
        """
        all_keys = set()
        for item in storage:
            all_keys.update(item['fields'].keys())

        final_out: Dict[str, torch.Tensor] = {}
        if not all_keys:
            return template_out

        max_len = max(
            [sum(chunk.shape[0] for chunk in item['fields'].get(next(iter(all_keys)), [])) if item['fields'] else 0
             for item in storage] + [1]
        )

        for key in all_keys:
            sample = None
            for item in storage:
                if key in item['fields'] and item['fields'][key]:
                    sample = item['fields'][key][0]
                    break
            if sample is None:
                continue

            padded_batches = []
            for item in storage:
                if key not in item['fields'] or not item['fields'][key]:
                    shape = (max_len, *sample.shape[1:])
                    padded_batches.append(torch.zeros(shape, dtype=sample.dtype, device=sample.device))
                    continue

                scores = torch.cat(item['scores'], dim=0)
                values = torch.cat(item['fields'][key], dim=0)

                # 按分数从高到低排序
                order = torch.argsort(scores, descending=True)
                values = values[order]

                if values.shape[0] < max_len:
                    pad = torch.zeros((max_len - values.shape[0], *values.shape[1:]),
                                      dtype=values.dtype, device=values.device)
                    values = torch.cat([values, pad], dim=0)
                else:
                    values = values[:max_len]

                padded_batches.append(values)

            final_out[key] = torch.stack(padded_batches, dim=0)

        # 最终有效 mask
        valid_masks = []
        for item in storage:
            total = sum(chunk.shape[0] for chunk in item['fields'].get(next(iter(all_keys)), [])) if item['fields'] else 0
            mask = torch.zeros(max_len, dtype=torch.bool, device=next(iter(final_out.values())).device)
            mask[:min(total, max_len)] = True
            valid_masks.append(mask)
        final_out['valid_mask'] = torch.stack(valid_masks, dim=0)

        return final_out

    def get_reference_points(self,
                             query: torch.Tensor,
                             transformation: torch.Tensor,
                             projection: torch.Tensor,
                             shape: torch.Size) -> torch.Tensor:
        """
        根据当前 query 的 3D 中心点，生成某个视角下的 2D reference points。

        功能：
            把 query 中心从 3D 空间映射到某个视角的特征图平面，
            得到归一化后的 (u, v)，供 deformable attention 采样使用。

        处理步骤：
            1. 如果提供 transformation，先把 query 从笛卡尔坐标变换到对应传感器坐标系
            2. 再转到球坐标（例如适配雷达）
            3. 使用 projection 投影到图像/特征图平面
            4. 除以深度项，得到实际像素坐标
            5. 再按 feature map 大小归一化到 [0, 1]
        """
        if transformation.any():
            # 先做坐标变换
            reference_points = torch.einsum(
                'bij,bkj->bki',
                transformation,
                torch.dstack((query[..., :3], torch.ones_like(query[..., 0])))
            )

            # 变换到球坐标
            r, phi, roh = cart2spher(
                reference_points[..., 0],
                reference_points[..., 1],
                reference_points[..., 2],
                degrees=True
            )
            reference_points = torch.dstack((r, phi, roh))
        else:
            reference_points = query

        # 再投影到该视角平面
        reference_points = torch.einsum(
            'bij,bkj->bki',
            projection,
            torch.dstack((reference_points[..., :3], torch.ones_like(reference_points[..., 0])))
        )

        # 透视除法
        mask = (reference_points[..., 2] != 0)

        u = reference_points[..., 0]
        u[mask] = reference_points[..., 0][mask] / reference_points[..., 2][mask]

        v = reference_points[..., 1]
        v[mask] = reference_points[..., 1][mask] / reference_points[..., 2][mask]

        # 按 feature map 尺寸归一化到 [0, 1]
        u = (u - 0) / (shape[:, 1].unsqueeze(1) - 0) * (1 - 0) + 0
        v = (v - 0) / (shape[:, 0].unsqueeze(1) - 0) * (1 - 0) + 0

        reference_points = torch.dstack((u, v))
        reference_points = torch.clip(reference_points, min=0.0, max=1.0)

        return reference_points

    def forward(self,
                batch: List[Dict[str, torch.Tensor]],
                shape: List[torch.Tensor],
                projection: List[Tuple[torch.Tensor, torch.Tensor]],
                out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播流程。

        输入：
            batch: 多视角多尺度特征
            shape: 每个视角原始特征图大小
            projection: 每个视角的投影矩阵和坐标变换矩阵
            out: 当前状态字典，至少包含 out['center']

        主流程：
            1. 初始化 query 和 query 位置编码
            2. 初始化 valid_mask，表示哪些 query 当前有效
            3. 对每一轮：
                a. 根据当前 center 计算各视角 reference points
                b. 用 MPFusion 融合多视角特征
                c. 用 head 输出分类/框
                d. 若是最后一轮，直接收集结果
                e. 若启用 query_pruning，则做 top-k 保留
                f. 若启用 early_exit，则把“高置信+稳定”的 query 提前收集并退出
                g. 剩余 query 进入下一轮
            4. 把所有轮收集到的 query 拼成最终输出
        """
        # self._ensure_runtime_attrs()

        B = out['center'].shape[0]

        # 初始 query: (B, N, d_model)
        query = self.query.unsqueeze(0).repeat(B, 1, 1)

        # 初始位置编码: (B, N, d_model)
        query_pos = self.query_embedding.weight.unsqueeze(0).repeat(B, 1, 1)

        valid_mask = None

        # 有效 query mask，初始全 True
        if self.query_pruning or self.early_exit:
            valid_mask = torch.ones((B, query.shape[1]), dtype=torch.bool, device=query.device)
            final_storage = [{'scores': [], 'fields': {}} for _ in range(B)]

        # 用于缓存最终输出（包含 early exit 的 query）

        exit_stats = []

        for i, (layer, head) in enumerate(zip(self.mpfusion.values(), self.heads)):
            # 保存上一轮输出，用于判断当前 query 是否稳定
            if self.query_pruning or self.early_exit:
                prev_out = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in out.items()}

            # 根据当前中心点，为每个视角生成参考点
            reference_points = [
                self.get_reference_points(out['center'][..., :3], p[0], p[1], s)
                for p, s in zip(projection, shape)
            ]

            # 多视角融合；query_mask 用于屏蔽 padding query
            if valid_mask is not None:
                query = layer(query, batch, reference_points, query_pos, query_mask=valid_mask)
            else:
                query = layer(query, batch, reference_points, query_pos)

            # 预测当前轮输出
            out = head(query, out)

            if self.query_pruning or self.early_exit:
                # 提取 query 分数
                scores = self._get_query_scores(out)

                is_last_iter = (i == self.i_iter - 1)
                if is_last_iter:
                    # 最后一轮：所有有效 query 都作为最终结果
                    self._append_to_final_storage(final_storage, out, scores, valid_mask)
                    break
            # -----------------------------
            # 1) 分阶段裁剪 query
            # -----------------------------
            # if self.query_pruning and i >= self.early_exit_from_iter:
            #     keep_k = self.stage_keep_counts[min(i, len(self.stage_keep_counts) - 1)]
            #
            #     # 在当前有效 query 中保留 top-k
            #     keep_idx, keep_valid_mask = self._masked_topk(scores, valid_mask, keep_k)
            #
            #     # query / query_pos / out / prev_out 全部同步裁剪
            #     query = self._batched_gather_tensor(query, keep_idx)
            #     query_pos = self._batched_gather_tensor(query_pos, keep_idx)
            #     out = self._batched_gather_out(out, keep_idx)
            #     prev_out = self._batched_gather_out(prev_out, keep_idx)
            #     scores = self._batched_gather_tensor(scores.unsqueeze(-1), keep_idx).squeeze(-1)
            #
            #     # 更新新的有效 mask
            #     valid_mask = keep_valid_mask

            # -----------------------------
            # 2) query early exit
            # -----------------------------
            if self.early_exit and i >= self.early_exit_iter:
                # 哪些 query 已经高置信且足够稳定，可以提前退出
                exit_mask = self._compute_exit_mask(out, prev_out, scores, valid_mask, i)

                # 统计当前迭代 early-exit 数量
                exited_per_batch = exit_mask.sum(dim=1)  # shape: (B,)
                exited_total = int(exited_per_batch.sum().item())  # 当前迭代整个 batch 退出总数

                # print(f"\n[Early Exit {i}], total={exited_total + B*100}, per_batch={(exited_per_batch + 100).tolist()}")
                print(f"\n[Early Exit {i}], total={exited_total}, per_batch={(exited_per_batch).tolist()}")

                # 把这些退出的 query 存到最终结果缓存
                if exit_mask.any():
                    self._append_to_final_storage(final_storage, out, scores, exit_mask)

                # continue_mask: 还要继续参加后续 refinement 的 query
                continue_mask = valid_mask & (~exit_mask)

                # 如果一个都不剩，直接返回最终结果
                if continue_mask.sum().item() == 0:
                    final_out = self._build_final_output(final_storage, out)
                    return final_out

                # 重新打包剩余 query，形成新的紧凑 batch
                cont_idx, cont_valid_mask = self._pack_continue_mask(continue_mask)

                query = self._batched_gather_tensor(query, cont_idx)
                query_pos = self._batched_gather_tensor(query_pos, cont_idx)
                out = self._batched_gather_out(out, cont_idx)

                # 更新有效 mask
                valid_mask = cont_valid_mask

        # 汇总最终结果

        if self.query_pruning or self.early_exit:
            final_out = self._build_final_output(final_storage, out)
            return final_out
        else:
            return out

def build_basempoptfusion(*args, **kwargs):
    return IMPFusion.from_config(*args, **kwargs)
