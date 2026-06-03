from __future__ import annotations  # noqa: F407

from collections import OrderedDict
from typing import Any, Dict

import torch
import torchvision

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

###新增
from torchvision.models.resnet import ResNet, Bottleneck

class ScaledResNet(ResNet):
    def __init__(self, block, layers, norm_layer=None, **kwargs):
        super().__init__(block=block, layers=layers, norm_layer=norm_layer, **kwargs)

def resnet152_x2(norm_layer=None) -> nn.Module:
    width_mult = 2
    layers = [3, 8, 36, 3]  # resnet152

    # stage planes 放大：64/128/256/512 -> ×2
    p1, p2, p3, p4 = [64 * width_mult, 128 * width_mult, 256 * width_mult, 512 * width_mult]

    model = ScaledResNet(block=Bottleneck, layers=layers, norm_layer=norm_layer)

    # stem 放大：conv1 输出和 inplanes 都要跟着变
    model.inplanes = 64 * width_mult
    model.conv1 = nn.Conv2d(3, 64 * width_mult, kernel_size=7, stride=2, padding=3, bias=False)
    model.bn1 = (norm_layer or nn.BatchNorm2d)(64 * width_mult)

    # 重新构造 4 个 stage（layer1..4）
    model.layer1 = model._make_layer(Bottleneck, p1, layers[0])
    model.layer2 = model._make_layer(Bottleneck, p2, layers[1], stride=2)
    model.layer3 = model._make_layer(Bottleneck, p3, layers[2], stride=2)
    model.layer4 = model._make_layer(Bottleneck, p4, layers[3], stride=2)

    return model

def resnet152_x4(norm_layer=None) -> nn.Module:
    width_mult = 4  # <- 这里
    layers = [3, 8, 36, 3]

    p1, p2, p3, p4 = [64 * width_mult, 128 * width_mult, 256 * width_mult, 512 * width_mult]

    model = ScaledResNet(block=Bottleneck, layers=layers, norm_layer=norm_layer)

    model.inplanes = 64 * width_mult
    model.conv1 = nn.Conv2d(3, 64 * width_mult, kernel_size=7, stride=2, padding=3, bias=False)
    model.bn1 = (norm_layer or nn.BatchNorm2d)(64 * width_mult)

    model.layer1 = model._make_layer(Bottleneck, p1, layers[0])
    model.layer2 = model._make_layer(Bottleneck, p2, layers[1], stride=2)
    model.layer3 = model._make_layer(Bottleneck, p3, layers[2], stride=2)
    model.layer4 = model._make_layer(Bottleneck, p4, layers[3], stride=2)

    return model

###新增


class BackboneBase(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 in_channels: int = 3,
                 multi_scale: int = 1,
                 channel_last: bool = True,
                 weights: OrderedDict[str, Any] = None,
                 **kwargs):
        """Base class for ResNet backbones with intermediate returns.

        Arguments:
            backbone: Backbone model to use. One of
                either ResNet18, ResNet34, ResNet50,
                ResNet101 or ResNet152.
            in_channels: Number of input feature channles.
                Must be provided if the number of input
                features is different from 3.
            multi_scale: Number of multiscale feature maps
                to return.
            channel_last: Channle format of the given input data.
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
        return_layers = {'layer' + str(i + 1): str(i + 1) for i in range(self._multi_scale)}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # Load custom weights
        if weights:
            self.load_state_dict(weights)

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
    def _to_channel_last(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return OrderedDict({k: v.movedim(1, -1) for k, v in batch.items()})

    def forward(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns the output of the forward pass for the selected backbone.

        Arguments:
            batch: Batch of input tensors with shape (B, H, W, C)
                for channel last or (B, C, H, W) for channel first.

        Returns:
            out: Output tensors ordered by LiFo.
                Returns the intermediate layer outputs
                of the ResNet backbone in reverse order,
                e.g. (layer3 out, layer2 out, layer1 out).
        """
        # Adjust channel format
        if self.channel_last:
            batch = batch.movedim(-1, 1)

        # Adjust input channels (B, C, H, W) -> (B, 3, H, W)
        out = self.adjustment_layer(batch)

        # Extract features
        out = self.body(out)

        # Adjust channel format
        if self.channel_last:
            out = self._to_channel_last(out)

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

        # Initialize parent class
        super().__init__(backbone, in_channels, multi_scale, weights=self.weights)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Backbone:  # noqa: F821
        return cls(**config)

    def _get_backbone(self, name: str, *args, **kwargs) -> nn.Module:

        ###新增
        if name.lower() in ["resnet152x2", "resnet152_x2"]:
            norm_layer = kwargs.get("norm_layer", None)
            if self.weights:
                raise ValueError("ResNet152x2 has no official pretrained weights. Set weights=''")
            return resnet152_x2(norm_layer=norm_layer)
        
        if name.lower() in ["resnet152x4", "resnet152_x4"]:
            norm_layer = kwargs.get("norm_layer", None)
            if self.weights:
                raise ValueError("ResNet152x4 has no official pretrained weights. Set weights=''")
            return resnet152_x4(norm_layer=norm_layer)

        ###新增

        # Get backbone model
        try:
            backbone = getattr(torchvision.models, name.lower())
        except AttributeError:
            backbone = getattr(torch.nn, name.lower())
        except Exception as e:
            raise e

        if not self.weights:
            return backbone(*args, **kwargs)

        # Get pretrained model weights
        try:
            # Load official weights
            weights = torchvision.models.get_weight(f"{name}_Weights.{self.weights}")
        except ValueError:
            # Load custom weights
            self.weights = torch.load(self.weights)
            return backbone(*args, **kwargs)
        except Exception as e:
            raise e
        else:
            self.weights = None

        return backbone(weights=weights, *args, **kwargs)

    @staticmethod
    def _get_norm_layer(name: str) -> nn.Module:
        try:
            return getattr(torchvision.ops, name)
        except AttributeError:
            return getattr(torch.nn, name)
        except Exception as e:
            raise e


def build_resnet(*args, **kwargs):
    return Backbone.from_config(*args, **kwargs)
