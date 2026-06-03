# inspect_model.py

import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_module_tree(model: nn.Module, max_depth=3):
    print("\n========== Module Tree ==========")

    for name, module in model.named_modules():
        depth = name.count(".")
        if depth > max_depth:
            continue

        indent = "  " * depth
        module_name = name if name else "[root]"
        print(f"{indent}{module_name}: {module.__class__.__name__}")


def inspect_torch_checkpoint(path: str):
    print(f"Loading PyTorch checkpoint: {path}")

    ckpt = torch.load(path, map_location="cpu")

    print("\n========== Checkpoint Type ==========")
    print(type(ckpt))

    # 情况 1：保存的是整个 nn.Module
    if isinstance(ckpt, nn.Module):
        model = ckpt
        print("\n========== Model Architecture ==========")
        print(model)

        total, trainable = count_params(model)
        print("\n========== Parameters ==========")
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,}")

        print_module_tree(model)
        return

    # 情况 2：checkpoint 是 dict
    if isinstance(ckpt, dict):
        print("\n========== Checkpoint Keys ==========")
        for k in ckpt.keys():
            print(k)

        # 常见字段名
        possible_state_dict_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "module",
        ]

        state_dict = None

        for key in possible_state_dict_keys:
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                print(f"\nFound state_dict under key: {key}")
                break

        # 有些文件本身就是 state_dict
        if state_dict is None:
            tensor_like_items = {
                k: v for k, v in ckpt.items()
                if torch.is_tensor(v)
            }
            if tensor_like_items:
                state_dict = ckpt

        if state_dict is None:
            print("\n没有找到可识别的 state_dict。")
            return

        print_state_dict_summary(state_dict)
        return

    print("\n无法识别该 checkpoint 格式。")


def print_state_dict_summary(state_dict):
    print("\n========== State Dict Summary ==========")

    total_params = 0
    grouped = defaultdict(list)

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue

        total_params += tensor.numel()

        # 按顶层模块分组
        top_module = name.split(".")[0]
        grouped[top_module].append((name, tuple(tensor.shape), tensor.numel()))

    print(f"Total tensor params: {total_params:,}")

    print("\n========== Layers / Tensors ==========")
    for module_name, items in grouped.items():
        print(f"\n[{module_name}]")
        for name, shape, numel in items:
            print(f"  {name:80s} shape={shape}, params={numel:,}")


def inspect_huggingface_model(model_path: str, trust_remote_code=False):
    from transformers import AutoConfig, AutoModel

    print(f"Loading Hugging Face model: {model_path}")

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    print("\n========== Config ==========")
    print(config)

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
        device_map=None,
    )

    print("\n========== Model Architecture ==========")
    print(model)

    total, trainable = count_params(model)

    print("\n========== Parameters ==========")
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    print_module_tree(model)

    print("\n========== Named Parameters ==========")
    for name, param in model.named_parameters():
        print(f"{name:80s} shape={tuple(param.shape)}, requires_grad={param.requires_grad}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径、checkpoint 文件路径，或 Hugging Face 模型名",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="auto",
        choices=["auto", "hf", "torch"],
        help="模型类型：auto / hf / torch",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="仅在你信任 Hugging Face 模型代码时开启",
    )

    args = parser.parse_args()
    model_path = args.model

    path = Path(model_path)

    if args.type == "hf":
        inspect_huggingface_model(model_path, args.trust_remote_code)
    elif args.type == "torch":
        inspect_torch_checkpoint(model_path)
    else:
        # auto 判断
        if path.exists() and path.is_file():
            inspect_torch_checkpoint(model_path)
        else:
            inspect_huggingface_model(model_path, args.trust_remote_code)


if __name__ == "__main__":
    main()