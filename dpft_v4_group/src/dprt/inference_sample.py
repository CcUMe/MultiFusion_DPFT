import argparse
from dprt.datasets import init
from dprt.datasets import load
from dprt.models import load as load_model
from evaluator import SimpleEvaluator
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed
import os
import os.path as osp
import torch.utils.data as data
from dprt.utils.visualize_all import visualize_to_stream

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(src: str, cfg: str, checkpoint: str, dst: str, sample_dir: str, split: str = None):
    """支持kradar_processed2/下任意样本（train/val/test）的推理"""
    config = load_config(cfg)
    print("=" * 60)
    # print("[单样本推理 - 支持全数据集]")
    print("=" * 60)
    # print(f"配置文件: {cfg}")
    print("\n=== 示例 1: BytesIO 流 ===")
    streams = visualize_to_stream(sample_dir, verbose=True)
    print(f"类型: {type(streams['camera'])}")
    print(f"大小: camera={len(streams['camera'].getvalue())} bytes")
    print(f"大小: radar ={len(streams['radar'].getvalue())} bytes")
    print(f"大小: lidar={len(streams['lidar'].getvalue())} bytes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT 单样本推理（支持全数据集）')
    parser.add_argument('--src', type=str, default="/mnt/disk1/xiaxue/datasets/kradar_processed2",
                        help="数据集根目录 e.g. /mnt/disk1/xiaxue/datasets/kradar_processed2")
    parser.add_argument('--cfg', type=str,default="/mnt/disk1/xiaxue/code/new/dpft_v4/config/kradar.json",
                        help="配置文件 e.g. /mnt/.../kradar.json")
    parser.add_argument('--checkpoint', type=str,default="/mnt/disk1/xiaxue/code/DPFT-main/DPFT-main/log/20250929-121010-276/checkpoints/20250929-121010-276_checkpoint_0103.pt",
                        help="模型checkpoint路径")
    parser.add_argument('--sample_dir', type=str, default="/mnt/disk1/xiaxue/datasets/kradar_processed2/test/12/00725_00689",
                        help="任意样本目录 e.g. /mnt/.../kradar_processed2/train/1/00033_00001")
    parser.add_argument('--split', type=str, default=None,
                        choices=['train', 'val', 'test'],
                        help="指定split（不填自动推断）")
    parser.add_argument('--dst', type=str, default="/mnt/disk1/xiaxue/code/new/dpft_v4/log/test_res",
                        help="输出目录")

    args = parser.parse_args()
    main(args.src, args.cfg, args.checkpoint, args.dst, args.sample_dir, args.split)
