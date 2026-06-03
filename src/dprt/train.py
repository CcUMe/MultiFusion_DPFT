import argparse
import datetime
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dprt.datasets import init as init_dataset
from dprt.datasets import load as load_dataset
from dprt.models import load_model
from dprt.models import build as build_model
from dprt.training import train as train_model
from dprt.utils.config import load_config, save_config
from dprt.utils.misc import set_seed


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
    parser.add_argument('--src', type=str, default='/mnt/disk1/xiaxue/datasets/kradar_processed2',
                        help="Path to the processed dataset folder.")
    parser.add_argument('--cfg', type=str, default='/mnt/disk1/xiaxue/code/new/dpft_v4/config/kradar.json',
                        help="Path to the configuration file.")
    parser.add_argument('--dst', type=str, default='/mnt/disk1/xiaxue/code/DPFT-main/DPFT-main/log',
                        help="Path to save the training log.")
    parser.add_argument('--checkpoint', type=str,default=None,
                        help="Path to a model checkpoint to resume training from.")
    args = parser.parse_args()


    main(src=args.src, cfg=args.cfg, dst=args.dst, checkpoint=args.checkpoint)
