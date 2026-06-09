import argparse
import datetime
import json
import os.path as osp

from collections import Counter

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
from dprt.datasets import init as init_dataset
from dprt.datasets import load as load_dataset
from dprt.models import load_model
from dprt.models import build as build_model
from dprt.training import train as train_model
from dprt.utils.config import load_config, save_config
from dprt.utils.misc import set_seed


def _shape_has_bbox(shape):
    points = shape.get("points") or []
    valid_points = [p for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
    return len(valid_points) >= 2


def _count_dataset_labels(dataset):
    categories = getattr(dataset, 'categories', {})
    aliases = getattr(dataset, 'label_aliases', {})
    samples = getattr(dataset, 'samples', [])
    counts = Counter()

    for sample in samples:
        json_path = sample.get('json')
        if json_path is None:
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data.get('shapes', []):
            label = str(shape.get('label', ''))
            label = aliases.get(label, label)
            class_idx = categories.get(label)
            if class_idx is None or class_idx <= 0 or not _shape_has_bbox(shape):
                continue
            counts[int(class_idx)] += 1

    return counts


def _print_dataset_label_stats(train_dataset, val_dataset):
    categories = getattr(train_dataset, 'categories', {})
    names = {
        int(idx): name
        for name, idx in categories.items()
        if isinstance(idx, int) and int(idx) > 0
    }
    train_counts = _count_dataset_labels(train_dataset)
    val_counts = _count_dataset_labels(val_dataset)

    print('\nDataset label statistics')
    print(f"{'Class':<22}{'Train':>10}{'Val':>10}{'Total':>10}")
    print('-' * 52)
    for class_idx in sorted(names):
        train_count = train_counts.get(class_idx, 0)
        val_count = val_counts.get(class_idx, 0)
        print(f"{names[class_idx]:<22}{train_count:>10}{val_count:>10}{train_count + val_count:>10}")
    print('-' * 52)
    print(f"{'Total':<22}{sum(train_counts.values()):>10}{sum(val_counts.values()):>10}{sum(train_counts.values()) + sum(val_counts.values()):>10}\n")


def main(src: str, cfg: str, dst: str, checkpoint: str = None):
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

    # Initialize training dataset
    train_dataset = init_dataset(dataset=config['dataset'], src=src, split='train', config=config)

    # Load training dataset
    train_loader = load_dataset(train_dataset, config=config)

    # Initialize validation dataset
    val_dataset = init_dataset(dataset=config['dataset'], src=src, split='val', config=config)

    # Load validation dataset
    val_loader = load_dataset(val_dataset, config=config)

    # Dataset label statistics are intentionally not printed during training.

    # Build model
    if checkpoint is not None:
        model, epoch, timestamp = load_model(checkpoint, config)
        epoch += 1
    else:
        model = build_model(config['model']['name'], config)

    # Save configuration (for logging)
    save_config(config, osp.join(dst, timestamp, 'config.json'))

    # Train model
    train_model(config)(model, train_loader, val_loader, epoch, timestamp, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT data preprocessing')
    parser.add_argument('--src', type=str, default='/mnt/disk1/zhangzhibin/dataset/LH_pairs_dataset_auto_v2',
                        help="Path to the processed dataset folder.")
    parser.add_argument('--cfg', type=str, default='/mnt/disk1/zhangzhibin/dpft_v4/config/la-tom-qwen14b.json',
                        help="Path to the configuration file.")
    parser.add_argument('--dst', type=str, default='/mnt/disk1/zhangzhibin/test/low-v2',
                        help="Path to save the training log.")
    parser.add_argument('--checkpoint', type=str,default=None,
                        help="Path to a model checkpoint to resume training from.")
    args = parser.parse_args()


    main(src=args.src, cfg=args.cfg, dst=args.dst, checkpoint=args.checkpoint)
