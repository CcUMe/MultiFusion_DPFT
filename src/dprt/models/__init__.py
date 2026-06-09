import os

from typing import Tuple, Optional, Type

import torch

from dprt.models.dprt import build_dprt
from dprt.models.dprt import DPRT   # 按你的项目实际导入路径改
from dprt.models.detectors import build_rgb_ir_query_detector
import json

def build(model: str, *args, **kwargs):
    if model == 'dprt':
        return build_dprt(*args, **kwargs)
    if model == 'rgb_ir_query_detector':
        return build_rgb_ir_query_detector(*args, **kwargs)


# def load_model(checkpoint: str, config_path: str, device="cpu") -> Tuple[torch.nn.Module, int, str]:
#     filename = os.path.splitext(os.path.basename(checkpoint))[0]
#     parts = filename.split("_")
#     if len(parts) < 3:
#         raise ValueError(f"非法 checkpoint 文件名: {filename}")
#
#     timestamp = parts[0]
#     epoch = int(parts[-1])
#
#     # 1) 读取配置文件
#     with open(config_path, "r", encoding="utf-8") as f:
#         config = json.load(f)
#
#     # 2) 先尝试读取 checkpoint
#     obj = torch.load(checkpoint, map_location=device)
#
#     # 3) 如果保存的是整个模型
#     if isinstance(obj, torch.nn.Module):
#         model = obj
#
#     # 4) 如果保存的是 state_dict
#     elif isinstance(obj, dict):
#         model = DPRT.from_config(config)
#         model.load_state_dict(obj, strict=True)
#
#     else:
#         raise TypeError(f"不支持的 checkpoint 类型: {type(obj)}")
#
#     model.to(device)
#     return model, epoch, timestamp

def load_model(checkpoint, config):
    filename = os.path.splitext(os.path.basename(checkpoint))[0]
    parts = filename.split("_")
    if len(parts) < 3:
        raise ValueError(f"非法 checkpoint 文件名: {filename}")

    timestamp = parts[0]
    epoch = int(parts[-1])

    obj = torch.load(checkpoint, map_location="cpu")

    if isinstance(obj, torch.nn.Module):
        model = obj

    elif isinstance(obj, dict):
        model_name = config.get("model", {}).get("name")
        model = build(model_name, config)
        if model is None:
            raise ValueError(f"Unsupported model type in config: {model_name!r}")
        state_dict = obj.get("model_state_dict", obj)
        model.load_state_dict(state_dict, strict=True)

    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    return model, epoch, timestamp