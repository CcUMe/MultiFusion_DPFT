from __future__ import annotations  # noqa: F407

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

from torch import nn
import torch.nn.functional as F

from dprt.models.layers.unary import Unary1d


class UnaryDetectionHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 num_reg_layers: int = 1,
                 num_cls_layers: int = 1,
                 bias: Optional[bool] = False,
                 dropout: float = 0.0,
                 channels_last: Optional[bool] = True,
                 **kwargs) -> None:
        # Initialize parent class
        super().__init__()

        # Initialize instance attributes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_reg_layers = num_reg_layers
        self.num_cls_layers = num_cls_layers
        self.bias = bias
        self.dropout = dropout
        self.channels_last = channels_last

        # Define activation functions
        self.activations = {
            'center': 'Identity',
            'size': 'ReLU',
            'angle': 'Tanh',
            'class': 'Identity'
        }

        # Initialize instance layers
        self.layers = nn.ModuleDict({
            'center_head': self._get_reg_branch(3),
            'size_head': self._get_reg_branch(3),
            'angle_head': self._get_reg_branch(2),
            'class_head': self._get_cls_branch(self.num_classes)
        })

        # Initialize activation functions
        self.activation_fn = nn.ModuleDict(
            {k: self._get_activation_fn(v) for k, v in self.activations.items()}
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> UnaryDetectionHead:  # noqa: F821
        return cls(
            config['in_channels'],
            config['num_classes'],
            config.get('num_reg_layers', 1),
            config.get('num_cls_layers', 1),
            config.get('bias', False),
            config.get('dropout', 0.0),
            config.get('channels_last', True)
        )

    def _get_activation_fn(self, name: str) -> nn.Module:
        if 'softmax' in name.lower() and self.channels_last:
            return getattr(nn, name)(dim=-1)
        if 'softmax' in name.lower():
            return getattr(nn, name)(dim=1)
        return getattr(nn, name)()

    def _get_cls_branch(self, out_channels: int) -> nn.Module:
        """Returns a sequence of unary layers

        Arguments:
            out_channels: Number of output channles for the last layer
                of the sequence.

        Returns:
            A sequence of unary layers.
        """
        cls_branch = []
        for _ in range(self.num_reg_layers - 1):
            cls_branch.append(Unary1d(self.in_channels, self.in_channels,
                                      bias=self.bias, channels_last=self.channels_last))
            cls_branch.append(nn.ReLU())
            cls_branch.append(nn.Dropout(self.dropout))

        cls_branch.append(Unary1d(self.in_channels, out_channels,
                                  bias=self.bias, channels_last=self.channels_last))

        return nn.Sequential(*cls_branch)

    def _get_reg_branch(self, out_channels: int) -> nn.Module:
        """Returns a sequence of unary layers with activation function

        Arguments:
            out_channels: Number of output channles for the last layer
                of the sequence.

        Returns:
            A sequence of unary layers with activation function, whereas
                the last layer does not have an activation function.
        """
        reg_branch = []
        for _ in range(self.num_reg_layers - 1):
            reg_branch.append(Unary1d(self.in_channels, self.in_channels,
                                      bias=self.bias, channels_last=self.channels_last))
            reg_branch.append(nn.ReLU())
            reg_branch.append(nn.Dropout(self.dropout))

        reg_branch.append(Unary1d(self.in_channels, out_channels,
                                  bias=self.bias, channels_last=self.channels_last))

        return nn.Sequential(*reg_branch)

    def forward(self, batch: torch.Tensor,
                ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Returns bounding box predictions given a feature tensor and reference points.

        Arguments:
            batch: Batched feature tensor with shape (B, N, in_channels).
            ref: Ordered dictionary that contains at least this entry:
                "center": Bounding box center coordinates of shape (B, N, 3).

        Returns:
            out: Ordered dictionary that contains these entries:
                "class": Bounding box class probabilities of shape (B, N, num_classes)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).
        """
        # Apply layers and activations
        iterator = zip(self.activation_fn.items(), self.layers.values())

        out = OrderedDict(
            {k: activation(layer(batch)) for (k, activation), layer in iterator}
        )

        # Add reference position to relative center position
        out['center'][..., :3] += ref['center'][..., :3]

        return out


class LinearDetectionHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 num_reg_layers: int = 1,
                 num_cls_layers: int = 1,
                 bias: Optional[bool] = False,
                 dropout: float = 0.0,
                 channels_last: Optional[bool] = True,
                 **kwargs) -> None:
        # Initialize parent class
        super().__init__()

        # Initialize instance attributes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_reg_layers = num_reg_layers
        self.num_cls_layers = num_cls_layers
        self.bias = bias
        self.dropout = dropout
        self.channels_last = channels_last

        # Define activation functions
        self.activations = {
            'center': 'Identity',
            'size': 'ReLU',
            'angle': 'Tanh',
            'class': 'Identity'
        }

        # Initialize instance layers
        self.layers = nn.ModuleDict({
            'center_head': self._get_reg_branch(3),
            'size_head': self._get_reg_branch(3),
            'angle_head': self._get_reg_branch(2),
            'class_head': self._get_cls_branch(self.num_classes)
        })

        # Initialize activation functions
        self.activation_fn = nn.ModuleDict(
            {k: self._get_activation_fn(v) for k, v in self.activations.items()}
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> LinearDetectionHead:  # noqa: F821
        return cls(
            config['in_channels'],
            config['num_classes'],
            config.get('num_reg_layers', 1),
            config.get('num_cls_layers', 1),
            config.get('bias', False),
            config.get('dropout', 0.0),
            config.get('channels_last', True)
        )

    def _get_activation_fn(self, name: str) -> nn.Module:
        if 'softmax' in name.lower() and self.channels_last:
            return getattr(nn, name)(dim=-1)
        if 'softmax' in name.lower():
            return getattr(nn, name)(dim=1)
        return getattr(nn, name)()

    def _get_cls_branch(self, out_channels: int) -> nn.Module:
        """Returns a sequence of linear layers

        Arguments:
            out_channels: Number of output channles for the last layer
                of the sequence.

        Returns:
            A sequence of linear layers.
        """
        cls_branch = []
        for _ in range(self.num_cls_layers - 1):
            cls_branch.append(nn.Linear(self.in_channels, self.in_channels, bias=self.bias))
            cls_branch.append(nn.ReLU())
            cls_branch.append(nn.Dropout(self.dropout))

        cls_branch.append(nn.Linear(self.in_channels, out_channels, bias=self.bias))

        return nn.Sequential(*cls_branch)

    def _get_reg_branch(self, out_channels: int) -> nn.Module:
        """Returns a sequence of linear layers with activation function

        Arguments:
            out_channels: Number of output channles for the last layer
                of the sequence.

        Returns:
            A sequence of linear layers with activation function, whereas
                the last layer does not have an activation function.
        """
        reg_branch = []
        for _ in range(self.num_reg_layers - 1):
            reg_branch.append(nn.Linear(self.in_channels, self.in_channels, bias=self.bias))
            reg_branch.append(nn.ReLU())
            reg_branch.append(nn.Dropout(self.dropout))

        reg_branch.append(nn.Linear(self.in_channels, out_channels, bias=self.bias))

        return nn.Sequential(*reg_branch)

    def forward(self, batch: torch.Tensor, ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            batch: Batched feature tensor with shape (B, N, in_channels).
            ref: Reference bounding box positions with shape (B, N, 3).

        Returns:
            out: Ordered dictionary that contains these entries:
                "class": Bounding box class probabilities of shape (B, N, num_classes)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).
        """
        # Apply layers and activations
        iterator = zip(self.activation_fn.items(), self.layers.values())

        out = OrderedDict(
            {k: activation(layer(batch)) for (k, activation), layer in iterator}
        )

        # Add reference position to relative center position
        out['center'][..., :3] += ref['center'][..., :3]

        return out


class BAGSLinearDetectionHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 groups=None,
                 num_reg_layers: int = 1,
                 num_cls_layers: int = 1,
                 bias: Optional[bool] = False,
                 dropout: float = 0.0,
                 channels_last: Optional[bool] = True,
                 **kwargs) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.groups = self._normalize_groups(groups, num_classes)
        self.num_reg_layers = num_reg_layers
        self.num_cls_layers = num_cls_layers
        self.bias = bias
        self.dropout = dropout
        self.channels_last = channels_last

        self.group_sizes = [len(group) + 1 for group in self.groups]
        self.num_bags_logits = sum(self.group_sizes)

        self.center_head = self._get_branch(3, self.num_reg_layers)
        self.size_head = self._get_branch(3, self.num_reg_layers)
        self.angle_head = self._get_branch(2, self.num_reg_layers)
        self.objectness_head = self._get_branch(1, self.num_cls_layers)
        self.bags_head = self._get_branch(self.num_bags_logits, self.num_cls_layers)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BAGSLinearDetectionHead:  # noqa: F821
        bags_config = config.get('bags', {})
        return cls(
            config['in_channels'],
            config['num_classes'],
            bags_config.get('groups', config.get('groups')),
            config.get('num_reg_layers', 1),
            config.get('num_cls_layers', 1),
            config.get('bias', False),
            config.get('dropout', 0.0),
            config.get('channels_last', True)
        )

    @staticmethod
    def _normalize_groups(groups, num_classes: int):
        if groups is None or groups == 'auto':
            if num_classes == 8:
                return [[3, 4, 5, 7], [2, 6], [1]]
            return [list(range(1, num_classes))]

        normalized = []
        for group in groups:
            normalized.append([int(class_idx) for class_idx in group])
        return normalized

    def _get_branch(self, out_channels: int, num_layers: int) -> nn.Module:
        branch = []
        for _ in range(num_layers - 1):
            branch.append(nn.Linear(self.in_channels, self.in_channels, bias=self.bias))
            branch.append(nn.ReLU())
            branch.append(nn.Dropout(self.dropout))

        branch.append(nn.Linear(self.in_channels, out_channels, bias=self.bias))
        return nn.Sequential(*branch)

    def _bags_to_class_scores(self,
                              objectness: torch.Tensor,
                              bags_logits: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(
            bags_logits.shape[:-1] + (self.num_classes, ),
            dtype=bags_logits.dtype,
            device=bags_logits.device
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

    def forward(self, batch: torch.Tensor, ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = OrderedDict()
        out['center'] = self.center_head(batch)
        out['size'] = torch.relu(self.size_head(batch))
        out['angle'] = torch.tanh(self.angle_head(batch))
        out['objectness'] = self.objectness_head(batch)
        out['bags_logits'] = self.bags_head(batch)

        out['center'][..., :3] += ref['center'][..., :3]
        out['class'] = self._bags_to_class_scores(out['objectness'], out['bags_logits'])

        return out


class BAGS2DDetectionHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 groups=None,
                 num_reg_layers: int = 1,
                 num_cls_layers: int = 1,
                 bias: Optional[bool] = False,
                 dropout: float = 0.0,
                 channels_last: Optional[bool] = True,
                 **kwargs) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.groups = BAGSLinearDetectionHead._normalize_groups(groups, num_classes)
        self.num_reg_layers = num_reg_layers
        self.num_cls_layers = num_cls_layers
        self.bias = bias
        self.dropout = dropout
        self.channels_last = channels_last

        self.group_sizes = [len(group) + 1 for group in self.groups]
        self.num_bags_logits = sum(self.group_sizes)

        self.layers = nn.ModuleDict({
            'box_head': self._get_branch(4, self.num_reg_layers),
            'objectness_head': self._get_branch(1, self.num_cls_layers),
            'bags_head': self._get_branch(self.num_bags_logits, self.num_cls_layers)
        })

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BAGS2DDetectionHead':
        bags_config = config.get('bags', {})
        return cls(
            config['in_channels'],
            config['num_classes'],
            bags_config.get('groups', config.get('groups')),
            config.get('num_reg_layers', 1),
            config.get('num_cls_layers', 1),
            config.get('bias', False),
            config.get('dropout', 0.0),
            config.get('channels_last', True)
        )

    def _get_branch(self, out_channels: int, num_layers: int) -> nn.Module:
        branch = []
        for _ in range(num_layers - 1):
            branch.append(nn.Linear(self.in_channels, self.in_channels, bias=self.bias))
            branch.append(nn.ReLU())
            branch.append(nn.Dropout(self.dropout))

        branch.append(nn.Linear(self.in_channels, out_channels, bias=self.bias))
        return nn.Sequential(*branch)

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x = x.clamp(min=eps, max=1.0 - eps)
        return torch.log(x / (1.0 - x))

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        half_w = w * 0.5
        half_h = h * 0.5
        return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)

    def _decode_boxes(self,
                      batch: torch.Tensor,
                      ref: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.layers['box_head'](batch)
        center = torch.sigmoid(out[..., :2] + self._inverse_sigmoid(ref['center'][..., :2]))
        size = torch.sigmoid(out[..., 2:])
        return torch.cat((center, size), dim=-1)

    def _bags_to_class_scores(self,
                              objectness: torch.Tensor,
                              bags_logits: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(
            bags_logits.shape[:-1] + (self.num_classes,),
            dtype=bags_logits.dtype,
            device=bags_logits.device
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

    def forward(self, batch: torch.Tensor, ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = OrderedDict(ref)

        out['boxes'] = self._decode_boxes(batch, ref)
        out['boxes_xyxy'] = torch.clamp(self._cxcywh_to_xyxy(out['boxes']), 0.0, 1.0)
        out['reference_points'] = ref['center'][..., :2]
        out['objectness'] = self.layers['objectness_head'](batch)
        out['bags_logits'] = self.layers['bags_head'](batch)
        out['class'] = self._bags_to_class_scores(out['objectness'], out['bags_logits'])

        return out


def build_detection_head(name: str, *args, **kwargs) -> nn.Module:
    if 'bags_2d' in name.lower():
        return BAGS2DDetectionHead.from_config(*args, **kwargs)

    if 'bags' in name.lower():
        return BAGSLinearDetectionHead.from_config(*args, **kwargs)

    if 'unary' in name.lower():
        return UnaryDetectionHead.from_config(*args, **kwargs)

    if 'linear' in name.lower():
        return LinearDetectionHead.from_config(*args, **kwargs)
