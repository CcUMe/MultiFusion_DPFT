#!/usr/bin/env python3
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torchvision

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dprt.models.backbones.resnet_custom import resnet101_custom


CUSTOM_INPUTS = {'camera_mono', 'ir_image', 'night_vision_image', 'radar_bev', 'radar_front', 'lidar_bev'}
STANDARD_ONLY_INPUTS = set()
STANDARD_BACKBONES = {
    'camera_mono': {'name': 'ResNet101', 'weights': 'IMAGENET1K_V2'},
    'ir_image': {'name': 'ResNet101', 'weights': 'IMAGENET1K_V2'},
    'night_vision_image': {'name': 'ResNet101', 'weights': 'IMAGENET1K_V2'},
    'radar_bev': {'name': 'ResNet50', 'weights': 'IMAGENET1K_V2'},
    'radar_front': {'name': 'ResNet50', 'weights': 'IMAGENET1K_V2'},
    'lidar_bev': {'name': 'ResNet50', 'weights': 'IMAGENET1K_V2'},
}
STANDARD_OUTPUTS = {
    'ResNet101': [256, 512, 1024, 2048],
    'ResNet50': [256, 512, 1024, 2048],
}


def get_block_depths(model):
    return [
        len(model.layer1),
        len(model.layer2),
        len(model.layer3),
        len(model.layer4),
    ]


def get_stage_specs(model):
    specs = []
    for layer_name in ('layer1', 'layer2', 'layer3', 'layer4'):
        layer = getattr(model, layer_name)
        first_block = layer[0]
        last_block = layer[-1]
        if hasattr(first_block, 'conv1') and hasattr(last_block, 'bn3'):
            bottleneck_width = int(first_block.conv1.out_channels)
            output_width = int(last_block.bn3.num_features)
        elif hasattr(first_block, 'conv1') and hasattr(last_block, 'bn2'):
            bottleneck_width = int(first_block.conv1.out_channels)
            output_width = int(last_block.bn2.num_features)
        else:
            raise RuntimeError(f'Unable to infer stage spec for {layer_name}')
        specs.append({
            'name': layer_name,
            'blocks': len(layer),
            'bottleneck_width': bottleneck_width,
            'output_width': output_width,
        })
    return specs


def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_params(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def describe_models():
    standard = torchvision.models.resnet101(weights=None)
    custom = resnet101_custom()
    return {
        'standard': {
            'name': 'ResNet101',
            'stem_width': int(standard.conv1.out_channels),
            'depths': get_block_depths(standard),
            'stage_specs': get_stage_specs(standard),
            'outputs': [spec['output_width'] for spec in get_stage_specs(standard)],
            'params': count_params(standard),
            'trainable_params': count_trainable_params(standard),
        },
        'custom': {
            'name': 'ResNet101-c',
            'stem_width': int(custom.conv1.out_channels),
            'depths': get_block_depths(custom),
            'stage_specs': get_stage_specs(custom),
            'outputs': [spec['output_width'] for spec in get_stage_specs(custom)],
            'params': count_params(custom),
            'trainable_params': count_trainable_params(custom),
        }
    }


def fpn_channels(input_channels, outputs, use_skiplink):
    return ([input_channels] + outputs) if use_skiplink else outputs


def replace_backbones(config, target):
    model = config['model']
    backbones = model.get('backbones', {})
    skiplinks = model.get('skiplinks', {})
    necks = model.get('necks', {})

    info = describe_models()
    custom_outputs = info['custom']['outputs']

    for key, backbone in backbones.items():
        if target == 'standard':
            standard_spec = STANDARD_BACKBONES.get(key)
            if standard_spec is None:
                raise ValueError(f'No standard backbone mapping defined for input: {key}')
            backbone['name'] = standard_spec['name']
            backbone['weights'] = standard_spec['weights']
            selected_outputs = STANDARD_OUTPUTS[standard_spec['name']]
        else:
            if key in CUSTOM_INPUTS:
                backbone['name'] = 'ResNet101-c'
                backbone['weights'] = None
                selected_outputs = custom_outputs
            elif key in STANDARD_ONLY_INPUTS:
                standard_spec = STANDARD_BACKBONES.get(key)
                backbone['name'] = standard_spec['name']
                backbone['weights'] = standard_spec['weights']
                selected_outputs = STANDARD_OUTPUTS[standard_spec['name']]
            else:
                raise ValueError(f'Unsupported input for backbone switching: {key}')

        if key in necks and necks[key].get('name', '').lower() == 'fpn':
            use_skiplink = bool(skiplinks.get(key, False))
            input_channels = int(backbone.get('in_channels', 3))
            necks[key]['in_channels_list'] = fpn_channels(input_channels, selected_outputs, use_skiplink)

    return info


def print_summary(cfg_path, target, info, config):
    standard = info['standard']
    custom = info['custom']
    param_delta = custom['params'] - standard['params']
    trainable_delta = custom['trainable_params'] - standard['trainable_params']

    print(f'Config: {cfg_path}')
    print(f'Switched backbone target: {target}')
    print('')
    print('ResNet101 -> ResNet101-c expansion summary')
    print(f"  Stem width: {standard['stem_width']} -> {custom['stem_width']}")
    print(f"  Depths: {standard['depths']} -> {custom['depths']}")
    print(f"  Stage outputs: {standard['outputs']} -> {custom['outputs']}")
    print('')
    print('Stage-by-stage width/depth change')
    for std_stage, custom_stage in zip(standard['stage_specs'], custom['stage_specs']):
        depth_ratio = custom_stage['blocks'] / std_stage['blocks']
        bottleneck_ratio = custom_stage['bottleneck_width'] / std_stage['bottleneck_width']
        output_ratio = custom_stage['output_width'] / std_stage['output_width']
        print(
            f"  {std_stage['name']}: blocks {std_stage['blocks']} -> {custom_stage['blocks']} "
            f"(x{depth_ratio:.2f}), bottleneck {std_stage['bottleneck_width']} -> {custom_stage['bottleneck_width']} "
            f"(x{bottleneck_ratio:.2f}), output {std_stage['output_width']} -> {custom_stage['output_width']} "
            f"(x{output_ratio:.2f})"
        )
    print('')
    print('Parameter expansion')
    print(
        f"  Total params: {standard['params']:,} ({standard['params'] / 1e6:.3f} M) -> "
        f"{custom['params']:,} ({custom['params'] / 1e6:.3f} M)"
    )
    print(
        f"  Added params: {param_delta:,} ({param_delta / 1e6:.3f} M), "
        f"x{custom['params'] / standard['params']:.3f}"
    )
    print(
        f"  Trainable params: {standard['trainable_params']:,} ({standard['trainable_params'] / 1e6:.3f} M) -> "
        f"{custom['trainable_params']:,} ({custom['trainable_params'] / 1e6:.3f} M)"
    )
    print(
        f"  Added trainable params: {trainable_delta:,} ({trainable_delta / 1e6:.3f} M)"
    )
    print('')
    print('Updated config entries')
    for key, backbone in config['model'].get('backbones', {}).items():
        print(f"  {key}.name = {backbone.get('name')}")
        print(f"  {key}.weights = {backbone.get('weights')}")
        if 'in_channels' in backbone:
            print(f"  {key}.in_channels = {backbone.get('in_channels')}")
        neck = config['model'].get('necks', {}).get(key, {})
        if neck.get('name', '').lower() == 'fpn':
            print(f"  {key}.fpn.in_channels_list = {neck.get('in_channels_list')}")


def main():
    parser = argparse.ArgumentParser('Switch backbone config between standard and visual-custom modes')
    parser.add_argument('--cfg', type=str,
                        default='/mnt/disk1/zhangzhibin/dpft_v4/config/la-tom-qwen14b.json',
                        help='Path to config json file.')
    parser.add_argument('--target', choices=['standard', 'custom'], default='custom',
                        help='standard restores the original ResNet101/ResNet50 mapping; custom upgrades only visual branches to ResNet101-c.')
    parser.add_argument('--dry-run', action='store_true', help='Print changes without writing the config file.')
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    updated = deepcopy(config)
    info = replace_backbones(updated, args.target)
    print_summary(cfg_path, args.target, info, updated)

    if args.dry_run:
        print('\nDry run only, config file not modified.')
        return

    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(updated, f, ensure_ascii=False, indent=4)
        f.write('\n')
    print(f'\nWrote updated config to: {cfg_path}')

if __name__ == '__main__':
    main()
