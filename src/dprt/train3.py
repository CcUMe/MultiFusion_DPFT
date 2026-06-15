import argparse
import datetime
import os.path as osp

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
from dprt.datasets import init as init_dataset
from dprt.datasets import load as load_dataset
from dprt.models import load_model
from dprt.models import build as build_model
from dprt.training import train as train_model
from dprt.utils.config import load_config, save_config
from dprt.utils.misc import set_seed


def _count_named_parameters(parameters):
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable = sum(parameter.numel() for _, parameter in parameters if parameter.requires_grad)
    frozen = total - trainable
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
    }


def parameter_counts(module, fusion_only=False):
    if module is None:
        return {'total': 0, 'trainable': 0, 'frozen': 0}

    named_parameters = list(module.named_parameters())
    if not fusion_only:
        return _count_named_parameters(named_parameters)

    fused_inputs = set(getattr(module, 'fuser_inputs', []) or [])
    available_inputs = set(getattr(module, 'available_inputs', []) or [])
    if fused_inputs and available_inputs:
        excluded_inputs = sorted(available_inputs - fused_inputs)
        excluded_prefixes = []
        for input_name in excluded_inputs:
            excluded_prefixes.extend([
                f'backbones.{input_name}.',
                f'necks.{input_name}.',
                f'embeddings.{input_name}.',
            ])
        named_parameters = [
            (name, parameter)
            for name, parameter in named_parameters
            if not any(name.startswith(prefix) for prefix in excluded_prefixes)
        ]

    return _count_named_parameters(named_parameters)


def print_parameter_counts(model, confidence=None, fusion_only=False):
    language_model = getattr(model, 'language_model', None)
    core_total = parameter_counts(model, fusion_only=fusion_only)
    language_counts = parameter_counts(language_model)

    core_counts = {
        key: core_total[key] - language_counts[key]
        for key in core_total
    }
    confidence_counts = parameter_counts(confidence)
    combined_counts = {
        key: core_counts[key] + language_counts[key] + confidence_counts[key]
        for key in core_counts
    }

    def fmt(count: int) -> str:
        return f"{count:>12,}  ({count / 1e6:.3f} M)"

    print('=' * 64)
    print(f"{'Module':<16} {'Total':>22} {'Trainable':>22} {'Frozen':>22}")
    print('-' * 64)
    print(
        f"{'DPRT Core':<16} {fmt(core_counts['total']):>22} "
        f"{fmt(core_counts['trainable']):>22} {fmt(core_counts['frozen']):>22}"
    )
    print(
        f"{'Language Model':<16} {fmt(language_counts['total']):>22} "
        f"{fmt(language_counts['trainable']):>22} {fmt(language_counts['frozen']):>22}"
    )
    print(
        f"{'Confidence':<16} {fmt(confidence_counts['total']):>22} "
        f"{fmt(confidence_counts['trainable']):>22} {fmt(confidence_counts['frozen']):>22}"
    )
    print('-' * 64)
    print(
        f"{'Combined':<16} {fmt(combined_counts['total']):>22} "
        f"{fmt(combined_counts['trainable']):>22} {fmt(combined_counts['frozen']):>22}"
    )


def main(src: str, cfg: str, dst: str, checkpoint: str = None, params_only: bool = False):
    """ Data preparation for subsequent model training or evaluation.

    Arguments:
        scr: Source directory path to the raw dataset folder.
        cfg: Path to the configuration file.
        dst: Destination directory to save the processed dataset files.
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]

    # Initialize start epoch
    epoch = 0

    # Load dataset configuration
    config = load_config(cfg)

    # Set global random seed
    set_seed(config['computing']['seed'])

    params_only = bool(params_only or config.get('params_only', False))

    if params_only:
        model = build_model(config['model']['name'], config)
        print_parameter_counts(model)
        return

    resume_state = None

    # Build model
    if checkpoint is not None:
        model, epoch, timestamp, resume_state = load_model(checkpoint, config)
        epoch += 1
    else:
        model = build_model(config['model']['name'], config)

    # Initialize training dataset
    train_dataset = init_dataset(dataset=config['dataset'], src=src, split='train', config=config)

    # Load training dataset
    train_loader = load_dataset(train_dataset, config=config)

    # Initialize validation dataset
    val_dataset = init_dataset(dataset=config['dataset'], src=src, split='val', config=config)

    # Load validation dataset
    val_loader = load_dataset(val_dataset, config=config)

    # Save configuration (for logging)
    save_config(config, osp.join(dst, timestamp, 'config.json'))

    # Train model
    train_model(config)(
        model, train_loader, val_loader, epoch, timestamp, dst,
        optimizer_state_dict=resume_state.get('optimizer_state_dict') if resume_state else None,
        scheduler_state_dict=resume_state.get('scheduler_state_dict') if resume_state else None,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT data preprocessing')
    parser.add_argument('--src', type=str, default='/mnt/disk1/zhangzhibin/dataset/LH_pairs_dataset_ir_labels',
                        help="Path to the processed dataset folder.")
    parser.add_argument('--cfg', type=str, default='/mnt/disk1/zhangzhibin/dpft_v4/config/la-tom-night.json',
                        help="Path to the configuration file.")
    parser.add_argument('--dst', type=str, default='/mnt/disk1/zhangzhibin/test/night',
                        help="Path to save the training log.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to a model checkpoint to resume training from.")
    parser.add_argument('--params-only', action='store_true',
                        help="Only build the model and print parameter counts.")
    args = parser.parse_args()


    main(src=args.src, cfg=args.cfg, dst=args.dst, checkpoint=args.checkpoint, params_only=args.params_only)
