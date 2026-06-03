import argparse

from dprt.datasets import prepare
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed


def main(src: str, cfg: str, dst: str):
    """ Data preparation for subsequent model training or evaluation.

    Arguments:
        scr: Source directory path to the raw dataset folder.
        cfg: Path to the configuration file.
        dst: Destination directory to save the processed dataset files.
    """
    # Load dataset configuration
    config = load_config(cfg)

    # Set global random seed
    set_seed(config['computing']['seed'])

    # Prepare dataset
    preperator = prepare(config['dataset'], config)
    preperator.prepare(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT data preprocessing')
    parser.add_argument('--src', type=str, default='/mnt/mobile_disk/kradar',
                        help="Path to the raw dataset folder.")
    parser.add_argument('--cfg', type=str, default='/mnt/disk1/xiaxue/code/new/dpft_v4/config/kradar.json',
                        help="Path to the configuration file.")
    parser.add_argument('--dst', type=str, default='/mnt/disk3/xiaxue/dataset/kradar_vlm',
                        help="Path to save the processed dataset.")
    args = parser.parse_args()

    main(src=args.src, cfg=args.cfg, dst=args.dst)
