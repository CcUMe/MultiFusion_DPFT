# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from __future__ import annotations  # noqa: F407

from typing import Any, Dict, Tuple

import torch
import numpy as np

from scipy.optimize import linear_sum_assignment
from torch import nn

from dprt.utils.bbox import get_box_corners
from dprt.utils.iou import giou3d


class HungarianAnassigner(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this,
    in general, there are more predictions than targets. In this case,
    we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 loss_weights: Dict[str, float] = None,
                 giou_weight: float = 1.0,
                 **kwargs):
        """Creates the matcher

        Arguments:
            loss_weights: Dictionary of loss weights. Mapping a
                model prediction name (in inputs) to a loss
                weight.
        """
        super().__init__()

        self.loss_weights = loss_weights
        self.giou_weight = giou_weight

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> HungarianAnassigner:  # noqa: F821
        loss_weights = config.get('loss_weights')
        return cls(
            loss_weights=loss_weights
        )

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Performs the matching

        The shape variable correspond to:
            B: Batch size
            N: Number of predicted objects
            M: Number of target objects
            C: Number of classes

        Arguments:
            outputs: This is a dict that contains at least these entries:
                "class": Bounding box class probabilities of shape (B, N, C)
                "center": Bounding box center coordinates of shape (B, N, C1).
                "size": Bounding box size values of shape (B, N, C2).
                "angle": Bounding box orientation values of shape (B, N, C3).

            targets: This is a dict of targets that contains at least these entries:
                "gt_class": Bounding box class probabilities of shape (B, M, C)
                "gt_center": Bounding box center coordinates of shape (B, M, C1).
                "gt_size": Bounding box size values of shape (B, M, C2).
                "gt_angle": Bounding box orientation values of shape (B, M, C3).

        Returns:
            A tuple of tensors (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                    with shape (B, M)
                - index_j is the indices of the corresponding selected targets (in order)
                    with shape (B, M)
        """
        with torch.no_grad():
            # Get output shape
            bs, num_queries = outputs["class"].shape[:2]

            # Get number of bounding boxes per batch
            sizes = [targets['gt_class'].shape[1]]

            # Get output device
            device = outputs["class"].device

            # We flatten to compute the cost matrices in a batch
            out_class = outputs["class"].flatten(0, 1)
            out_center = outputs["center"].flatten(0, 1)
            out_size = outputs["size"].flatten(0, 1)
            out_angle = outputs["angle"].flatten(0, 1)

            gt_class = targets["gt_class"].flatten(0, 1)
            gt_center = targets["gt_center"].flatten(0, 1)
            gt_size = targets["gt_size"].flatten(0, 1)
            gt_angle = targets["gt_angle"].flatten(0, 1)

            gt_ids = torch.argmax(gt_class, dim=-1)

            # Compute the classification cost
            cost_class = -out_class[:, gt_ids]

            # Compute the L1 cost between boxes
            cost_center = torch.cdist(out_center, gt_center, p=1)
            cost_size = torch.cdist(out_size, gt_size, p=1)
            cost_angle = torch.cdist(out_angle, gt_angle, p=1)

            # Compute the giou cost betwen boxes
            out_angle = torch.atan2(outputs["angle"][..., 0], outputs["angle"][..., 1])
            gt_angle = torch.atan2(targets["gt_angle"][..., 0], targets["gt_angle"][..., 1])
            out_corners = get_box_corners(outputs["center"], outputs["size"], out_angle)
            gt_corners = get_box_corners(targets["gt_center"], targets["gt_size"], gt_angle)
            cost_giou = -giou3d(out_corners, gt_corners)

            # Final cost matrix
            class_weight = self.loss_weights.get(
                'total_class',
                self.loss_weights.get('bags_class', self.loss_weights.get('class', 1.0))
            )
            C = class_weight * cost_class \
                + self.loss_weights.get('center', 1.0) * cost_center \
                + self.loss_weights.get('size', 1.0) * cost_size \
                + self.loss_weights.get('angle', 1.0) * cost_angle \
                + self.giou_weight * cost_giou

            # Reconstruct original shape
            C = C.view(bs, num_queries, -1).cpu()

            # Match predictions and ground truth
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            i, j = zip(*indices)

            index_i = torch.from_numpy(np.stack(i)).to(device=device)
            index_j = torch.from_numpy(np.stack(j)).to(device=device)
            return index_i, index_j


# 简化的匈牙利分配器，直接处理边界框坐标
class SimpleHungarianAssigner(nn.Module):
    def __init__(self, loss_weights=None):
        super().__init__()
        self.loss_weights = loss_weights or {
            'center': 1.0,
            'size': 1.0,
            'angle': 1.0,
            'class': 1.0,
            'giou': 1.0
        }

    def forward(self, outputs, targets):
        """
        执行匹配操作

        参数:
            outputs: 包含字典的列表，每个字典包含"center"、"size"、"class"和"angle"键
            targets: 包含字典的列表，每个字典包含"center"、"size"、"gt_class"和"angle"键

        返回:
            索引元组(index_i, index_j)
        """
        with torch.no_grad():
            batch_size = min(len(outputs), len(targets))
            device = outputs[0]["class"].device

            all_index_i = []
            all_index_j = []

            # 遍历每个样本
            for batch_idx in range(batch_size):
                # 获取当前样本的预测和目标
                out_center = outputs[batch_idx]["center"]  # (num_queries, 3)
                out_size = outputs[batch_idx]["size"]  # (num_queries, 3)
                out_class = outputs[batch_idx]["class"]  # (num_queries, num_classes)
                out_angle = outputs[batch_idx]["angle"]  # (num_queries, 2)

                gt_center = targets[batch_idx]["gt_center"]  # (num_targets, 3)
                gt_size = targets[batch_idx]["gt_size"]  # (num_targets, 3)
                gt_class = targets[batch_idx]["gt_class"]  # (num_targets, num_classes)
                gt_angle = targets[batch_idx]["gt_angle"]  # (num_targets, 2)

                num_targets = gt_center.shape[0]

                # 如果没有目标，跳过匹配
                if num_targets == 0:
                    # 创建空的匹配索引
                    index_i = torch.zeros(0, dtype=torch.long, device=device)
                    index_j = torch.zeros(0, dtype=torch.long, device=device)
                    all_index_i.append(index_i)
                    all_index_j.append(index_j)
                    continue

                # 获取真实标签的类别索引
                gt_ids = torch.argmax(gt_class, dim=-1)

                # 计算各项成本
                # 计算中心点之间的L1成本
                cost_center = torch.cdist(out_center, gt_center, p=1)

                # 计算尺寸之间的L1成本
                cost_size = torch.cdist(out_size, gt_size, p=1)

                # 计算角度之间的L1成本
                cost_angle = torch.cdist(out_angle, gt_angle, p=1)

                # 计算分类成本
                cost_class = -out_class[:, gt_ids]
                # 计算GIoU成本
                # 将角度的sin和cos转换为角度值
                out_angle_rad = torch.atan2(out_angle[..., 0], out_angle[..., 1])
                gt_angle_rad = torch.atan2(gt_angle[..., 0], gt_angle[..., 1])

                # 为get_box_corners函数调整维度
                # 添加batch维度，使其变为 (1, num_queries, 3) 和 (1, num_targets, 3)
                out_center_batch = out_center.unsqueeze(0)  # (1, num_queries, 3)
                out_size_batch = out_size.unsqueeze(0)  # (1, num_queries, 3)
                out_angle_rad_batch = out_angle_rad.unsqueeze(0)  # (1, num_queries)

                gt_center_batch = gt_center.unsqueeze(0)  # (1, num_targets, 3)
                gt_size_batch = gt_size.unsqueeze(0)  # (1, num_targets, 3)
                gt_angle_rad_batch = gt_angle_rad.unsqueeze(0)  # (1, num_targets)

                # 计算边界框角点
                out_corners = get_box_corners(out_center_batch, out_size_batch, out_angle_rad_batch)
                gt_corners = get_box_corners(gt_center_batch, gt_size_batch, gt_angle_rad_batch)

                # 计算GIoU成本
                cost_giou = -giou3d(out_corners, gt_corners)
                cost_giou = cost_giou.squeeze(0)  # 移除batch维度，得到 (num_queries, num_targets)

                # 最终成本矩阵
                C = self.loss_weights.get('center', 1.0) * cost_center \
                    + self.loss_weights.get('size', 1.0) * cost_size \
                    + self.loss_weights.get('angle', 1.0) * cost_angle \
                    + self.loss_weights.get('class', 1.0) * cost_class \
                    + self.loss_weights.get('giou', 1.0) * cost_giou

                # 处理可能的无效数值（NaN或inf）
                # 将NaN替换为0，将inf替换为一个大数
                C = torch.nan_to_num(C, nan=0.0, posinf=1e6, neginf=-1e6)

                # 确保成本矩阵中的所有值都在合理范围内
                C = torch.clamp(C, min=-1e6, max=1e6)

                # 转换为numpy进行线性分配
                C_np = C.cpu().numpy()

                # 检查矩阵是否包含有效的数值
                if not np.isfinite(C_np).all():
                    # 如果仍然包含无效数值，使用一个简单的默认成本矩阵
                    C_np = np.ones_like(C_np)  # 使用全1矩阵作为默认值

                # 匈牙利算法匹配
                try:
                    index_i_np, index_j_np = linear_sum_assignment(C_np)
                except ValueError as e:
                    # 如果线性分配失败，创建空的匹配索引
                    index_i_np = np.array([], dtype=np.int64)
                    index_j_np = np.array([], dtype=np.int64)

                # 转换回torch张量
                index_i = torch.from_numpy(index_i_np).to(device=device)
                index_j = torch.from_numpy(index_j_np).to(device=device)

                all_index_i.append(index_i)
                all_index_j.append(index_j)

        return all_index_i, all_index_j


def build_anassigner(name: str, *args, **kwargs):
    if 'hungarian' in name.lower():
        return HungarianAnassigner.from_config(*args, **kwargs)
