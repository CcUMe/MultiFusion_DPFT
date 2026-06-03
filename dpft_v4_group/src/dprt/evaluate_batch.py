import argparse
import os
import torch

from dprt.datasets import init, load
from dprt.evaluation import evaluate
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed

# 切换到 file_system 共享策略，避免文件描述符不足导致的 ancdata 错误
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def run_once(src, cfg, ckpt_path, dst):
    # 加载配置并设置随机种子
    config = load_config(cfg)
    set_seed(config['computing']['seed'])
    # 初始化测试集与加载器
    test_dataset = init(dataset=config['dataset'], src=src, split='test', config=config)
    test_loader = load(test_dataset, config=config)
    # 执行评估
    evaluate(config)(ckpt_path, test_loader, dst)

def main():
    parser = argparse.ArgumentParser('Batch evaluate checkpoints')
    parser.add_argument('--src',       type=str, required=True, help='测试集数据目录')
    parser.add_argument('--cfg',       type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint',type=str, required=True,
                        help='单次评估权重路径，需包含 "{epoch}" 占位')
    parser.add_argument('--dst',       type=str, required=True, help='评估结果输出目录')
    parser.add_argument('--start_epoch', type=int, default=None, help='起始 epoch')
    parser.add_argument('--end_epoch',   type=int, default=None, help='结束 epoch')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # 如果指定了范围，则批量评估
    if args.start_epoch is not None and args.end_epoch is not None:
        for epoch in range(args.start_epoch, args.end_epoch + 1):
            ckpt_path = args.checkpoint.format(epoch=f"{epoch:04d}")
            if not os.path.isfile(ckpt_path):
                print(f"跳过，未找到文件: {ckpt_path}")
                continue
            epoch_dst = os.path.join(args.dst, f"epoch_{epoch:04d}")
            os.makedirs(epoch_dst, exist_ok=True)
            print(f"评估 epoch {epoch} -> {ckpt_path}")
            run_once(args.src, args.cfg, ckpt_path, epoch_dst)
    else:
        # 仅评估单个 checkpoint
        run_once(args.src, args.cfg, args.checkpoint, args.dst)

if __name__ == '__main__':
    main()