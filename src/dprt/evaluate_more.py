import argparse

from torch.utils.data import Subset

from dprt.datasets import init
from dprt.datasets import load
from dprt.evaluation import evaluate
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(src: str, cfg: str, checkpoint_dir: str, dst: str, start_epoch: int, end_epoch: int):
    """ Data preparation for subsequent model training or evaluation.
    Arguments:
        src: Source directory path to the raw dataset folder.
        cfg: Path to the configuration file.
        checkpoint_dir: Directory containing checkpoint files.
        dst: Destination directory to save the processed dataset files.
        start_epoch: Starting epoch number (e.g., 60).
        end_epoch: Ending epoch number (e.g., 70).
    """
    # Load dataset configuration
    config = load_config(cfg)

    # Set global random seed
    set_seed(config['computing']['seed'])

    # Initialize test dataset
    test_dataset = init(dataset=config['dataset'], src=src, split='test', config=config)

    # 只取前 100 个样本
    # num_samples = 10
    # test_dataset = Subset(test_dataset, list(range(min(num_samples, len(test_dataset)))))

    # Load test dataset
    test_loader = load(test_dataset, config=config)

    # Loop through checkpoints from start_epoch to end_epoch
    for epoch in range(start_epoch, end_epoch + 1):
        # 构造checkpoint文件路径，根据您的命名格式调整
        checkpoint_path = os.path.join(checkpoint_dir, f'*_checkpoint_{epoch:04d}.pt')

        # 如果checkpoint_dir包含完整路径模式，则直接使用
        import glob
        checkpoint_files = glob.glob(checkpoint_path)

        if not checkpoint_files:
            print(f"Warning: Checkpoint for epoch {epoch} not found at {checkpoint_path}")
            continue

        checkpoint = checkpoint_files[0]  # 取第一个匹配的文件
        print(f"\nEvaluating checkpoint: {checkpoint}")

        # Evaluate model at checkpoint
        evaluate(config)(checkpoint, test_loader, dst)
        print(f"Completed evaluation for epoch {epoch}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT evaluation for multiple checkpoints')

    parser.add_argument('--src', type=str, default='/mnt/disk1/zhangzhibin/k_radar_v1.1',
                        help="Path to the processed dataset folder.")

    parser.add_argument('--cfg', type=str, default='/home/yangqilin/code/dpft_v4/config/kradar.json',
                        help="Path to the configuration file.")

    parser.add_argument('--checkpoint_dir', type=str,
                        default='/mnt/disk1/yangqilin/dpft/log/20260413-170614-206/checkpoints',
                        help="Directory containing checkpoint files.")

    parser.add_argument('--dst', type=str, default='/mnt/disk1/yangqilin/dpft/log',
                        help="Path to save the processed dataset.")

    parser.add_argument('--start_epoch', type=int, default=79,
                        help="Starting epoch number for evaluation.")

    parser.add_argument('--end_epoch', type=int, default=79,
                        help="Ending epoch number for evaluation.")

    args = parser.parse_args()

    main(src=args.src, cfg=args.cfg, checkpoint_dir=args.checkpoint_dir,
         dst=args.dst, start_epoch=args.start_epoch, end_epoch=args.end_epoch)