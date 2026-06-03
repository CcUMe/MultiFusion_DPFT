import argparse
import datetime
import os.path as osp
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
from collections import Counter

from dprt.datasets import init as init_dataset
from dprt.datasets import load as load_dataset
from dprt.models import load_model as load_model
from dprt.models import build as build_model
from dprt.training import train as train_model
from dprt.utils.config import load_config, save_config
from dprt.utils.misc import set_seed

import torch
import numpy as np
from torch.utils.data import Subset
torch.multiprocessing.set_sharing_strategy('file_system')

def print_model_params(model):
    """统计并打印模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print("=" * 45)
    print(f"  Total params    : {total:>12,}  ({total / 1e6:.3f} M)")
    print(f"  Trainable params: {trainable:>12,}  ({trainable / 1e6:.3f} M)")
    print(f"  Frozen params   : {frozen:>12,}  ({frozen / 1e6:.3f} M)")
    print("=" * 45)


def _iter_detection_label_paths(dataset):
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        for index in dataset.indices:
            sample_paths = base_dataset.dataset_paths[index]
            if 'label' in sample_paths:
                yield sample_paths['label']
        return

    for sample_paths in getattr(dataset, 'dataset_paths', []):
        if 'label' in sample_paths:
            yield sample_paths['label']


def print_class_summary(dataset, config):
    categories = config.get('data', {}).get('categories', {})
    class_names = {idx: name for name, idx in categories.items() if idx >= 0}
    counts = Counter()
    samples = 0
    labeled_samples = 0

    for label_path in _iter_detection_label_paths(dataset):
        samples += 1
        raw_label = np.load(label_path)
        if raw_label.size == 0:
            continue
        label = dataset.dataset.get_detection_label(torch.from_numpy(raw_label)) \
            if isinstance(dataset, Subset) else dataset.get_detection_label(torch.from_numpy(raw_label))
        if label['gt_class'].numel() == 0:
            continue
        class_indices = label['gt_class'].argmax(dim=1).tolist()
        counts.update(index - 1 for index in class_indices)
        labeled_samples += 1

    total = sum(counts.values())
    print("=" * 64)
    print("Training dataset class summary")
    print(f"Samples: {samples}")
    print(f"Samples with objects after FoV filtering: {labeled_samples}")
    print(f"Objects: {total}")
    print("-" * 64)
    if total == 0:
        print("No objects found in the training dataset.")
    else:
        print(f"{'Class':<24} {'Index':>8} {'Count':>12} {'Ratio':>10}")
        for class_idx in sorted(class_names):
            count = counts.get(class_idx, 0)
            ratio = count / total * 100
            print(f"{class_names[class_idx]:<24} {class_idx:>8} {count:>12} {ratio:>9.2f}%")
    print("=" * 64)


def main(src: str, cfg: str, dst: str, checkpoint: str = None):
    """ Data preparation for subsequent model training or evaluation.

    Arguments:
        src: Source directory path to the raw dataset folder.
        cfg: Path to the configuration file.
        dst: Destination directory to save the processed dataset files.
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]

    # Initialize start epoch
    epoch = 0

    # Load dataset configuration
    config = load_config(cfg)
    print(
        "CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"config device={config['computing']['device']}"
    )

    # Set global random seed
    set_seed(config['computing']['seed'])

    # Initialize training dataset
    train_dataset = init_dataset(dataset=config['dataset'], src=src, split='train', config=config)

    # Load training dataset
    train_loader = load_dataset(train_dataset, config=config)
    print_class_summary(train_loader.dataset, config)

    # Initialize validation dataset
    val_dataset = init_dataset(dataset=config['dataset'], src=src, split='val', config=config)

    # Load validation dataset
    val_loader = load_dataset(val_dataset, config=config)

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
    parser.add_argument('--src', type=str, default='/mnt/disk1/zhangzhibin/dataset/kradar/',
                        help="Path to the processed dataset folder.")
    parser.add_argument('--cfg', type=str, default='/home/yangqilin/code/dpft_v4/config/la-tom.json',
                        help="Path to the configuration file.")
    parser.add_argument('--dst', type=str, default='/mnt/disk1/yangqilin/dpft/log',
                        help="Path to save the training log.")
    parser.add_argument('--checkpoint', type=str,default=None,
                        help="Path to a model checkpoint to resume training from.")
    args = parser.parse_args()

    main(src=args.src, cfg=args.cfg, dst=args.dst, checkpoint=args.checkpoint)
