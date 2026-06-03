from __future__ import annotations  # noqa: F407

from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

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
                          query_positions: torch.Tensor = None) -> torch.Tensor:
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
        out = self.self_attn(query=q, key=k, value=query, need_weights=False)[0]

        # Apply dropout
        out = query + self.dropout1(out)

        # Apply normalization
        if self.norm:
            out = self.norm1(out)

        return out

    def forward_cross_attn(self,
                           query: torch.Tensor,
                           batch: Dict[str, torch.Tensor],
                           reference_points: torch.Tensor,
                           query_positions: torch.Tensor = None) -> torch.Tensor:
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
                query_positions: torch.Tensor = None) -> torch.Tensor:
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
        out = self.forward_self_attn(query=query, query_positions=query_positions)

        # Cross attention: Cross attend to multi level features
        out = self.forward_cross_attn(query=out, batch=batch,
                                      reference_points=reference_points,
                                      query_positions=query_positions)

        # FFN: Propagate attended features
        out = self.forward_ffn(query=out)

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
                query_positions: torch.Tensor) -> torch.Tensor:
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
                    query_positions
            )
            # print(f"After i={i}: queries[..., {i}].shape={queries[..., i].shape}")

        # print(f"Final queries.shape={queries.shape}")
        # Fuse multi perspective query features
        out = self.reduce(query, queries, query_positions)

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
                 **kwargs):
        """Iterative Multi-Perspective Fusion Transformer.

        Arguments:
            i_iter: Number of fusion and output refinement iterations.
            m_views: Number of perspective views to query from.
            d_model: Hidden feature (channel) dimension.
            n_queries: Number of queries fed to the fuser.
            n_levels: Number of feature levels for each view.
            n_heads: Number of attention heads for each view.
            n_points: Number of sampling points per attention head and
                per feature level for each view.
            q_init: Query feature initialization method.
            reduction: Reduction mode to fuse the queries of multiple views.
                One of either mean, max or cross-attn.
            head: Output head of the model. Used to generate the input
                for the next iteration and the final output.
        """
        # Initialize base class
        super().__init__()

        # Initialize instance attributes
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

        self.q_init = getattr(nn.init, q_init)

        if head is None:
            head = nn.Identity()

        # Initialize fusion layers
        self.mpfusion = nn.ModuleDict({
            'fusion' + str(i):
            MPFusion(self.m_views, self.d_model, self.d_ffn, self.n_levels,
                     self.n_heads, self.n_points, self.ffn_layer, self.activation,
                     self.dropout, self.norm, self.reduction)
            for i in range(self.i_iter)
        })

        # Initialize detection heads
        self.heads = self._get_clones(head, self.i_iter)

        # Initialize query positional embedding
        self.query_embedding = nn.Embedding(self.n_queries, self.d_model)

        # Initialize queries
        query = torch.empty((self.n_queries, self.d_model))
        self.query = nn.Parameter(query)

        self.reset_parameters()
        #新增投影层
        self.projection = ProjectionHead(self.d_model, 2560)

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> IMPFusion:  # noqa: F821
        return cls(**config, **kwargs)

    @staticmethod
    def _get_clones(module: nn.Module, n: int) -> nn.Module:
        """Retruns a module list of n cloned modules.

        Arguments:
            module: Modules to clone.
            n: Number of clones.

        Returns:
            A module list of n clones of the given module.
        """
        return nn.ModuleList([deepcopy(module) for i in range(n)])

    def reset_parameters(self) -> None:
        """Initialize query weights."""
        self.q_init(self.query)

    def get_reference_points(self,
                             query: torch.Tensor,
                             transformation: torch.Tensor,
                             projection: torch.Tensor,
                             shape: torch.Size) -> torch.Tensor:
        """Returns the query reference points in the feature space given a projection.

        Projects the query points (X, Y, Z) to the feature space (u, v)
        given a (4, 4) transformation matrix.

        [uw]   [p11 p12 p13 p14] [X]
        [vw] = [p21 p22 p23 p24] [Y]
        [ w]   [p31 p32 p33 p34] [Z]
        [ 1]   [0   0   0   1  ] [1]

        Arguments:
            query:  Query points with shape (B, N, 3).
            transformation: Transformation matrx given as (4, 4) homogeneous
                transformation matrix. If a transformation matrix is provided,
                the transformation is applied first (in cartesian space) before
                the reference points are transformed into spherical coordinates.
            projection: Projection matrx given as (4, 4) homogeneous
                transformation matrix. Projects the reference points into
                the sensor space.
            shape: Feature map shape with shape (B, 2).

        Returns:
            reference_points: Reference points with shape
                (B, N, 2) where 2 is ordered by H, W
        """
        if transformation.any():
            # Apply transformation to query points (T @ Q^T)
            reference_points = torch.einsum(
                'bij,bkj->bki',
                transformation,
                torch.dstack((query[..., :3], torch.ones_like(query[..., 0])))
            )

            # Convert reference points from cartesian to spherical coordinates
            r, phi, roh = cart2spher(
                reference_points[..., 0],
                reference_points[..., 1],
                reference_points[..., 2],
                degrees=True
            )

            reference_points = torch.dstack((r, phi, roh))

        else:
            reference_points = query

        # Get reference points in the feature map space (P @ Q^T)
        reference_points = torch.einsum(
            'bij,bkj->bki',
            projection,
            torch.dstack((reference_points[..., :3], torch.ones_like(reference_points[..., 0])))
        )

        # Scale reference points with the w value
        mask = (reference_points[..., 2] != 0)

        # Width index (pixel)
        u = reference_points[..., 0]
        u[mask] = reference_points[..., 0][mask] / reference_points[..., 2][mask]

        # Height index (pixel)
        v = reference_points[..., 1]
        v[mask] = reference_points[..., 1][mask] / reference_points[..., 2][mask]

        # Scale reference points according to the feature map size
        u = (u - 0) / (shape[:, 1].unsqueeze(1) - 0) * (1 - 0) + 0
        v = (v - 0) / (shape[:, 0].unsqueeze(1) - 0) * (1 - 0) + 0

        # Reduce reference points to 2d projection
        reference_points = torch.dstack((u, v))

        # Clip values to account for numerical issues
        reference_points = torch.clip(reference_points, min=0.0, max=1.0)

        return reference_points

    def forward(self,
                batch: List[Dict[str, torch.Tensor]],
                shape: List[torch.Tensor],
                projection: List[Tuple[torch.Tensor, torch.Tensor]],
                out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns an iteratively fused and refined output prediction.

        Arguments:
            batch: List of ordered dictionaries mapping a level to a tensor.
                The list has length m_views and each dict has length n_levels.
            shape: List of tensors representing the raw data input shapes.
                The list has length m_views and the tensors have shape (B, 2).
            projection: List of tuples, each containing two tensors representing
                homogeneous transformation matrices with shape (4, 4). The list
                has length m_views.
            out: Ordered dictionary of tensors that contains at least this entry:
                "center": Bounding box center coordinates of shape (B, N, 3).

        Returns:
            out: Ordered dictionary of tensors that contains these entries:
                "class": Bounding box class probabilities of shape (B, N, num_classes)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).
        """
        # Get batch size
        B = out['center'].shape[0]

        # Adjust query dimensions (N, d_model) -> (B, N, d_model)
        query = self.query.unsqueeze(0).repeat(B, 1, 1)

        # Get query positional embedding values (B, N, d_model)
        query_pos = self.query_embedding.weight.unsqueeze(0).repeat(B, 1, 1)

        for layer, head in zip(self.mpfusion.values(), self.heads):
            # Calculate reference points for each feature map
            # (4,400,2)
            reference_points = [
                self.get_reference_points(out['center'][..., :3], p[0], p[1], s)
                for p, s in zip(projection, shape)
            ]

            # Query features from multiple perspectives
            query = layer(query, batch, reference_points, query_pos)

            # Apply head to query features
            out = head(query, out)

        #投影最后结果
        # projected_query = self.projection(query)
        return out


def build_mpfusion(*args, **kwargs):
    return IMPFusion.from_config(*args, **kwargs)
