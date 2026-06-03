import argparse
from dprt.datasets import init
from dprt.datasets import load
from dprt.evaluation import evaluate
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed
import os
import torch.utils.data as data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def find_sample_index(test_dataset, sample_path, src):
    """
    快速查找样本索引

    Args:
        test_dataset: 数据集对象
        sample_path: 样本的完整路径
        src: 数据集根目录

    Returns:
        样本在数据集中的索引，如果未找到则返回 None
    """
    sample_identifier = os.path.relpath(sample_path, os.path.join(src, 'test'))
    print(f"🔍 Looking for: {sample_identifier}")

    # 使用 dataset_paths 属性快速查找（不加载数据）
    if hasattr(test_dataset, 'dataset_paths'):
        print(f"  Total samples: {len(test_dataset.dataset_paths)}")

        for idx, sample_dict in enumerate(test_dataset.dataset_paths):
            # sample_dict 是一个字典，包含所有文件路径
            # 检查任一路径是否包含目标标识
            for file_path in sample_dict.values():
                if sample_identifier in file_path or sample_path in file_path:
                    return idx

    return None


def main(src: str, cfg: str, checkpoint: str, dst: str, sample_path: str = None):
    """
    评估主函数

    Args:
        src: 数据集路径
        cfg: 配置文件路径
        checkpoint: 模型检查点路径
        dst: 输出目录
        sample_path: 可选，单个样本的路径
    """
    config = load_config(cfg)
    set_seed(config['computing']['seed'])

    # 初始化数据集
    test_dataset = init(dataset=config['dataset'], src=src, split='test', config=config)

    # 如果指定了样本路径，查找并过滤为单个样本
    if sample_path is not None:
        sample_idx = find_sample_index(test_dataset, sample_path, src)

        if sample_idx is None:
            raise ValueError(f"❌ Sample not found: {sample_path}")

        print(f"✓ Found at index: {sample_idx}")
        test_dataset = data.Subset(test_dataset, [sample_idx])
        print(f"✓ Testing single sample\n")
    else:
        print(f"✓ Testing {len(test_dataset)} samples\n")

    # 加载数据
    test_loader = load(test_dataset, config=config)
    os.makedirs(dst, exist_ok=True)

    # 运行标准评估
    print("Running evaluation...")
    evaluate(config)(checkpoint, test_loader, dst)

    print(f"\n✅ Evaluation completed!")
    print(f"📂 Results saved to: {dst}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DPRT Evaluation - Single Sample Support')

    parser.add_argument('--src', type=str,
                        default='/mnt/6feca051-3223-4732-996b-597606847bb2/xiaxue/datasets/kradar_processed2',
                        help="Dataset path")

    parser.add_argument('--cfg', type=str,
                        default='/mnt/6feca051-3223-4732-996b-597606847bb2/xiaxue/code/new/dpft_v4/config/kradar.json',
                        help="Config file path")

    parser.add_argument('--checkpoint', type=str,
                        default='/mnt/6feca051-3223-4732-996b-597606847bb2/xiaxue/code/DPFT-main/DPFT-main/log/20250929-121010-276/checkpoints/20250929-121010-276_checkpoint_0189.pt',
                        help="Checkpoint path")

    parser.add_argument('--dst', type=str,
                        default='/mnt/6feca051-3223-4732-996b-597606847bb2/xiaxue/code/new/dpft_v4/log',
                        help="Output path")

    parser.add_argument('--sample_path', type=str, default=None,
                        help="Specific sample path for single sample evaluation")

    args = parser.parse_args()

    main(src=args.src, cfg=args.cfg, checkpoint=args.checkpoint,
         dst=args.dst, sample_path=args.sample_path)
