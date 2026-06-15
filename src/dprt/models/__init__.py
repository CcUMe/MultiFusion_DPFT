import os

from typing import Tuple, Optional, Type

import torch

from dprt.models.dprt import build_dprt
from dprt.models.dprt import DPRT   # 按你的项目实际导入路径改
from dprt.models.detectors import build_rgb_ir_query_detector
from dprt.utils.config import get_active_inputs
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



def _inactive_state_prefixes_from_config(config):
    model_config = config.get("model", {})
    available_inputs = list(model_config.get("inputs") or [])
    active_inputs = set(get_active_inputs(model_config))
    inactive_inputs = [input_name for input_name in available_inputs if input_name not in active_inputs]
    prefixes = []
    for input_name in inactive_inputs:
        prefixes.extend([
            f'backbones.{input_name}.',
            f'necks.{input_name}.',
            f'embeddings.{input_name}.',
        ])
    return prefixes

def load_model(checkpoint, config):
    obj = torch.load(checkpoint, map_location="cpu")
    if not isinstance(obj, dict) or 'model_state_dict' not in obj:
        raise ValueError(
            'Checkpoint format is outdated. Please restart training with the new full-checkpoint format.'
        )

    model_name = config.get("model", {}).get("name")
    model = build(model_name, config)
    if model is None:
        raise ValueError(f"Unsupported model type in config: {model_name!r}")

    checkpoint_excluded_prefixes = obj.get('excluded_state_prefixes') or []
    current_excluded_prefixes = _inactive_state_prefixes_from_config(config)
    allowed_missing_prefixes = list(dict.fromkeys(checkpoint_excluded_prefixes + current_excluded_prefixes))
    strict = not allowed_missing_prefixes
    missing_keys, unexpected_keys = model.load_state_dict(obj['model_state_dict'], strict=strict)
    if unexpected_keys:
        raise ValueError(f'Unexpected checkpoint keys: {unexpected_keys}')
    if allowed_missing_prefixes:
        invalid_missing = [
            key for key in missing_keys
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if invalid_missing:
            raise ValueError(
                'Checkpoint is missing unexpected model weights: ' + ', '.join(invalid_missing[:20])
            )
    epoch = int(obj.get('epoch', 0))
    timestamp = obj.get('timestamp')
    if not timestamp:
        filename = os.path.splitext(os.path.basename(checkpoint))[0]
        parts = filename.split("_")
        if len(parts) < 3:
            raise ValueError(f"非法 checkpoint 文件名: {filename}")
        timestamp = parts[0]

    resume_state = {
        'optimizer_state_dict': obj.get('optimizer_state_dict'),
        'scheduler_state_dict': obj.get('scheduler_state_dict'),
    }
    return model, epoch, timestamp, resume_state
