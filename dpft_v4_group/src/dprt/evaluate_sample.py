import argparse
from dprt.datasets import init
from dprt.datasets import load
from dprt.evaluate_sample import SimpleEvaluator
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed
import os
import os.path as osp
import torch.utils.data as data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def main(src: str, cfg: str, checkpoint: str, dst: str, sample_dir: str = None):
    """Data preparation for subsequent model training or evaluation."""
    config = load_config(cfg)
    print("=" * 60)
    print("[数据集路径信息]")
    print("=" * 60)
    print(f"配置文件路径: {cfg}")
    print(f"数据集源目录: {src}")

    # Set global random seed
    set_seed(config['computing']['seed'])
    print("[设置随机种子]")

    # Initialize test dataset
    test_dataset = init(dataset=config['dataset'], src=src, split='test', config=config)
    print(f"[初始化测试数据集] len={len(test_dataset)}")

    # 只推理单个样本
    if sample_dir is not None:
        target_dir = osp.normpath(sample_dir)
        sample_idx = None

        # 非序列模式：datasetpaths 是 list[dict(modality -> filepath)]
        if hasattr(test_dataset, 'datasetpaths') and isinstance(test_dataset.datasetpaths, list):
            for idx, sp_dict in enumerate(test_dataset.datasetpaths):
                # 取任意一个文件路径（比如 ra.npy, mono.jpg 等）
                any_path = next(iter(sp_dict.values()))
                sp_dir = osp.normpath(osp.dirname(any_path))
                if sp_dir == target_dir:
                    sample_idx = idx
                    print(f"[找到样本] idx={sample_idx}, dir={sp_dir}, example_file={any_path}")
                    break

        # 如果没找到或 sequential=True，报错提示
        if sample_idx is None:
            print(f"[错误] 未找到样本目录: {target_dir}")
            print("调试信息:")
            print("  datasetpaths type:", type(test_dataset.datasetpaths))
            if hasattr(test_dataset, 'datasetpaths'):
                first_key = list(test_dataset.datasetpaths.keys())[0] if isinstance(test_dataset.datasetpaths,
                                                                                    dict) else \
                test_dataset.datasetpaths[0]
                print("  第一条样本示例:", first_key)
            raise ValueError(f"Sample not found: {sample_dir}")

        # Subset 只保留这个样本
        test_dataset = data.Subset(test_dataset, [sample_idx])
        print(f"[单样本模式] 只测试 idx={sample_idx}")

    # Load test dataset (现在loader里只有1个batch)
    test_loader = load(test_dataset, config=config)
    print("[加载测试DataLoader]")

    # Evaluate model at checkpoint
    evaluate(config)(checkpoint, test_loader, dst)
    print("[评估完成]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT data preprocessing')
    parser.add_argument('--src', type=str, default='/mnt/disk1/xiaxue/datasets/kradar_processed2',
                        help="Path to the processed dataset folder.")
    parser.add_argument('--cfg', type=str, default='/mnt/disk1/xiaxue/code/new/dpft_v4/config/kradar.json',
                        help="Path to the configuration file.")
    parser.add_argument('--dst', type=str, default='/mnt/disk1/xiaxue/code/DPFT-main/DPFT-main/log',
                        help="Path to save the training log.")
    parser.add_argument('--checkpoint', type=str,
                        default="/mnt/disk1/xiaxue/code/DPFT-main/DPFT-main/log/20250929-121010-276/checkpoints/20250929-121010-276_checkpoint_0103.pt",
                        help="Path to a model checkpoint to resume training from.")
    parser.add_argument('--sample_dir', type=str, default=None,
                        help="Path to specific test sample directory, e.g. /mnt/.../test/1/00284_00252")

    args = parser.parse_args()
    main(src=args.src, cfg=args.cfg, checkpoint=args.checkpoint, dst=args.dst, sample_dir=args.sample_dir)
