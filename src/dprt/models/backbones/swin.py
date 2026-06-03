from __future__ import annotations  # noqa: F407

from collections import OrderedDict
from typing import Any, Dict

import torch
import torchvision
import os
import re
from torch import nn
from torchvision.models.swin_transformer import SwinTransformer
from torchvision.models._utils import IntermediateLayerGetter


class BackboneBase(nn.Module):
    def __init__(self,
                 backbone: SwinTransformer,
                 in_channels: int = 3,
                 multi_scale: int = 1,
                 channel_last: bool = True,
                 weights: OrderedDict[str, Any] = None,
                 **kwargs):
        """Base class for SwinTransformer backbones with intermediate returns.

        Arguments:
            backbone: Backbone model to use. SwinTransformer
                variant (e.g. Swin_T).
            in_channels: Number of input feature channels.
                Must be provided if the number of input
                features is different from 3.
            multi_scale: Number of multiscale feature maps
                to return.
            channel_last: Channel format of the given input data.
                True if the input is given in channel last format,
                False otherwise.
            weights: Model state as ordered dictionary given to
                load pretrained weights.
        """
        # Initialize base class
        super().__init__()

        # Set instance properties
        self.in_channels = in_channels
        self.multi_scale = multi_scale
        self.channel_last = channel_last

        # Add adjustment layer to match the input channels
        if in_channels == 3:
            self.adjustment_layer = nn.Identity()
        else:
            self.adjustment_layer = nn.Conv2d(self.in_channels, out_channels=3, kernel_size=(1, 1),
                                              stride=1, padding=0, bias=False)

        # Determine intermediate features to return
        if weights is not None:
            new_sd=OrderedDict()
            for k,v in weights.items():
                new_sd[k.replace('layer.','').replace('features','')]=v
            backbone.load_state_dict(new_sd,strict=True)
        return_layers = {str(i): str((i // 2) + 1) for i in range(1, self._multi_scale * 2, 2)}
        self.body = IntermediateLayerGetter(backbone.features, return_layers=return_layers)

        # Load custom weights
        if weights:
            if isinstance(weights, dict) and 'model' in weights:
                state_dict = weights['model']
            else: 
                state_dict = weights
            self.load_state_dict(state_dict,strict=False)

    @property
    def multi_scale(self):
        return self._multi_scale

    @multi_scale.setter
    def multi_scale(self, value):
        if 5 > value < 0:
            raise ValueError(
                f"The number of multi scale feature maps "
                f"to retrun has to be in the range of [1, 4] "
                f"but a number of {value} feature maps were "
                f"requested."
            )
        self._multi_scale: int = value

    @staticmethod
    def _to_channel_first(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return OrderedDict({k: v.movedim(-1, 1) for k, v in batch.items()})

    def forward(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns the output of the forward pass for the selected backbone.

        Arguments:
            batch: Batch of input tensors with shape (B, H, W, C)
                for channel last or (B, C, H, W) for channel first.

        Returns:
            out: Output tensors ordered by LiFo.
                Returns the intermediate layer outputs
                of the backbone in reverse order,
                e.g. (layer3 out, layer2 out, layer1 out).
        """
        # Adjust channel format
        if self.channel_last:
            batch = batch.movedim(-1, 1)

        # Adjust input channels (B, C, H, W) -> (B, 3, H, W)
        out = self.adjustment_layer(batch)

        # Extract features, returns (B, H, W, C)
        out = self.body(out)

        # Adjust channel format
        if not self.channel_last:
            out = self._to_channel_first(out)

        return out


class Backbone(BackboneBase):
    def __init__(self,
                 name: str,
                 weights: str = '',
                 norm_layer: str = None,
                 in_channels: int = 3,
                 multi_scale: int = 1,
                 **kwargs):
        """Backbone wrapper class for the base backbones.

        Arguments:
            name: Name of the specific backbone type.
            weights: Pretrained model weights.
            norm_layer: Normalization layer passed to the backbone.
            multi_scale: Number of multiscale feature maps
                to return.
        """
        # Initialize instance attributes
        self.weights = weights

        # Initialize instance modules
        if norm_layer is not None:
            norm_layer = self._get_norm_layer(norm_layer)
        backbone = self._get_backbone(name, norm_layer=norm_layer)
        if isinstance(self.weights, str):
            self.weights = None

        # Initialize parent class
        super().__init__(backbone, in_channels, multi_scale, weights=self.weights)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Backbone:  # noqa: F821
        return cls(**config)

    def _get_backbone(self, name: str, *args, **kwargs) -> nn.Module:
         # ===== 私有模型优先处理 =====
        if name.lower() == "my_swin_s":
          from .my_swin_s import my_swin_s, Swin_S_Weights
          weights_enum = Swin_S_Weights.verify(self.weights) if self.weights else None
          norm_layer_val = kwargs.pop("norm_layer", None)
          return my_swin_s(weights=weights_enum, norm_layer=norm_layer_val, **kwargs)
    # ============================
        # Get backbone model
        try:
            backbone_factory = getattr(torchvision.models, name.lower())
        except AttributeError:
            backbone_factory = getattr(torch.nn, name.lower())

        if not self.weights:
            return backbone_factory(*args, **kwargs)

        # Get pretrained model weights
        if isinstance(self.weights, str) and re.match(r'^[A-Z]', self.weights):
            try:
                enum_name = re.sub(r'[^a-zA-Z0-9]', '_', name).title() + '_Weights'
                weights_enum_cls = getattr(torchvision.models, enum_name)
                weights = weights_enum_cls.verify(self.weights)
                return backbone_factory(weights=weights, *args, **kwargs)
            except(AttributeError, ValueError):
                print(f"预训练权重枚举未找到或无效: {self.weights}，将尝试从给定路径加载权重文件。")
        if not os.path.isfile(self.weights):
            raise FileNotFoundError(f"The given weight path '{self.weights}' does not exist.")
        ckpt =torch.load(self.weights, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        model=backbone_factory(*args, **kwargs)
        missing, unexpected =map_load_state_dict(state_dict,strict=True)
        if len(missing) == 0 or len(unexpected) == 0:
            print(f"模型权重加载成功: {self.weights} strict=True")
        else:
            print(f"模型权重加载存在缺失或多余参数，尝试 strict=False 重新加载: {self.weights}")
            model.load_state_dict(state_dict, strict=False)
        return model

    @staticmethod
    def _get_norm_layer(name: str) -> nn.Module:
        try:
            return getattr(torchvision.ops, name)
        except AttributeError:
            return getattr(torch.nn, name)
        except Exception as e:
            raise e


def build_swin(*args, **kwargs):
    return Backbone.from_config(*args, **kwargs)
