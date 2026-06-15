import argparse
import glob
import os
import os.path as osp

from dprt.datasets import init
from dprt.datasets import load
from dprt.evaluation import evaluate
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def _resolve_checkpoint(checkpoint: str | None) -> str:
    if not checkpoint:
        raise ValueError('Please provide --checkpoint or set a valid default checkpoint path in evaluate.py.')

    if osp.isfile(checkpoint):
        return checkpoint

    if not osp.isdir(checkpoint):
        raise FileNotFoundError(f'Checkpoint path does not exist: {checkpoint}')

    candidates = []
    best_path = osp.join(checkpoint, 'checkpoints', 'best.pt')
    if osp.isfile(best_path):
        return best_path

    search_roots = [
        osp.join(checkpoint, 'checkpoints', '*_checkpoint_*.pt'),
        osp.join(checkpoint, '*_checkpoint_*.pt'),
    ]
    for pattern in search_roots:
        candidates.extend(glob.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f'No checkpoint file found under directory: {checkpoint}. '
            'Expected best.pt or *_checkpoint_*.pt.'
        )

    return sorted(candidates)[-1]


def main(src: str, cfg: str, checkpoint: str, dst: str):
    """ Data preparation for subsequent model training or evaluation.

    Arguments:
        src: Source directory path to the raw dataset folder.
        cfg: Path to the configuration file.
        dst: Destination directory to save the processed dataset files.
    """
    # Load dataset configuration
    config = load_config(cfg)
    print("=" * 60)
    print("[数据集路径信息]")
    print("=" * 60)
    print(f"配置文件路径: {cfg}")
    print(f"数据集源目录: {src}")
    checkpoint = _resolve_checkpoint(checkpoint)
    print(f"评估权重路径: {checkpoint}")

    # Set global random seed
    set_seed(config['computing']['seed'])

    # Initialize test dataset
    test_dataset = init(dataset=config['dataset'], src=src, split='test', config=config)

    # Load test dataset
    test_loader = load(test_dataset, config=config)

    # Evaluate model at checkpoint
    evaluate(config)(checkpoint, test_loader, dst)


import torch.utils.data as data

# # 指定特定样本
# def main(src: str, cfg: str, checkpoint: str, dst: str, sample_path: str = None):
#     """ Data preparation for subsequent model training or evaluation.
#
#     Arguments:
#         src: Source directory path to the raw dataset folder.
#         cfg: Path to the configuration file.
#         checkpoint: Path to model checkpoint.
#         dst: Destination directory to save the processed dataset files.
#         sample_path: Path to specific sample to test (optional).
#     """
#     # Load dataset configuration
#     config = load_config(cfg)
#
#     # Set global random seed
#     set_seed(config['computing']['seed'])
#
#     # Initialize test dataset
#     test_dataset = init(dataset=config['dataset'], src=src, split='test', config=config)
#
#     # 如果指定了样本路径，只测试该样本
#     if sample_path is not None:
#         # 从路径中提取样本标识（例如 "1/00182_00150"）
#         sample_identifier = os.path.relpath(sample_path, os.path.join(src, 'test'))
#
#         print(f"Looking for sample: {sample_identifier}")
#
#         # 在数据集中查找匹配的样本
#         sample_idx = None
#         for idx in range(len(test_dataset)):
#             # 根据你的数据集类实际结构调整这里
#             # 常见属性名: samples, file_list, data_paths 等
#             if hasattr(test_dataset, 'samples'):
#                 item_path = test_dataset.samples[idx]
#             elif hasattr(test_dataset, 'file_list'):
#                 item_path = test_dataset.file_list[idx]
#             else:
#                 # 尝试通过 __getitem__ 获取
#                 item = test_dataset[idx]
#                 item_path = item.get('path', '') if isinstance(item, dict) else ''
#
#             # 检查路径是否匹配
#             if sample_identifier in str(item_path) or sample_path in str(item_path):
#                 sample_idx = idx
#                 print(f"Found sample at index: {idx}")
#                 break
#
#         if sample_idx is None:
#             raise ValueError(f"Sample not found: {sample_path}")
#
#         # 只保留这一个样本
#         test_dataset = data.Subset(test_dataset, [sample_idx])
#         print(f"Testing single sample: {sample_identifier}")
#     else:
#         print(f"Testing full dataset: {len(test_dataset)} samples")
#
#     # Load test dataset
#     test_loader = load(test_dataset, config=config)
#
#     # Evaluate model at checkpoint
#     evaluate(config)(checkpoint, test_loader, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT evaluation')
    parser.add_argument('--src', type=str, default='/mnt/disk1/zhangzhibin/dataset/LH_pairs_dataset_auto_v2',
                        help="Path to the processed dataset folder.")
    parser.add_argument('--cfg', type=str, default='/mnt/disk1/zhangzhibin/dpft_v4/config/la-tom-light.json',
                        help="Path to the configuration file.")
    parser.add_argument('--dst', type=str, default='/mnt/disk1/zhangzhibin/test/low-v2-eval',
                        help="Path to save the training log.")
    parser.add_argument('--checkpoint', type=str, default='/mnt/disk1/zhangzhibin/test/light/20260614-115352-428',
                        help="Path to a checkpoint .pt file or an experiment directory containing checkpoints.")
    args = parser.parse_args()


    main(src=args.src, cfg=args.cfg, checkpoint=args.checkpoint, dst=args.dst)
