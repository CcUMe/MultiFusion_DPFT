import argparse
import copy
import os.path

from tqdm import tqdm
from dprt.utils.draw import open_kradar_label_v1_1
from dprt.datasets import init_v2 as init
from dprt.datasets import load_v2 as load
from dprt.evaluation import evaluate_v2
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed

from dprt.models import load_v2 as load_model
from typing import Any, Callable, Dict, Iterable, List
import torch
from dprt.utils.data import decollate_batch
import matplotlib.image as mpimg
from dprt.utils.visu import visu_camera_data
from dprt.utils.project_boxes import get_camera_calibration_ss
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from dprt.utils.util_calib import show_projected_point_cloud
from dprt.utils.util_calib import get_matrices_from_dict_calib
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from dprt.utils.draw import open_kradar_label_v1_1, open_kradar_img, visu_kradar_img, open_kradar_label_v2_1
from dprt.utils.visu import visu_camera_data
import os
from mmdet3d.core.bbox import LiDARInstance3DBoxes
# 旧版 (v1.0.0rc4) 的正确导入路径
from mmdet3d.core.visualizer import show_multi_modality_result
from mmdet3d.core.bbox import CameraInstance3DBoxes
import mmcv
import numpy as np
import torch
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

# 25.12.25 画不同天气下的预测框

def _dict_to(data: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    # return {k: v.to(device) for k, v in data.items()}
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in data.items()}

def read_img(path):
    img = cv2.imread(path)
    return img

def convert_to_mmdet3d_format(centers, sizes, angles):
    """
    将字典列表格式的真值框转换为 mmdet3d 的 LiDARInstance3DBoxes 对象。

    Args:
        ground_truth_list: List[Dict], 每个字典包含 x, y, z, length, width, height, angle 等键

    Returns:
        bboxes_3d: LiDARInstance3DBoxes 对象
    """
    box_list = []
    for center, size, angle in zip(centers, sizes, angles):
        # 提取 7 个关键参数
        # 注意: mmdet3d 的 LiDAR 坐标系通常定义框的顺序为 [x, y, z, dx, dy, dz, yaw]
        # 且 yaw 通常是绕 Z 轴旋转
        box = [
            center[0]+2.54,
            center[1]-0.3,
            center[2]-0.7,
            size[0],  # dx (x方向长度)
            size[1],  # dy (y方向长度)
            size[2],  # dz (z方向高度)
            -torch.atan2(angle[0] , angle[1] )
        ]
        box_list.append(box)
    if len(box_list) == 0:
        return None
    # 1. 转换为 Tensor
    box_tensor = torch.tensor(box_list, dtype=torch.float32)
    # 2. 封装为 LiDARInstance3DBoxes
    # origin=(0.5, 0.5, 0.5): 假设你的 x,y,z 是几何中心
    # 如果你的 z 是底部中心，请改为 (0.5, 0.5, 0.0)
    bboxes_3d = LiDARInstance3DBoxes(box_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))
    return bboxes_3d


def show_combined_result(img,
                         gt_bboxes,
                         pred_bboxes,
                         proj_mat,
                         out_dir,
                         filename,
                         gt_color=(0, 255, 0),  # 绿色 (BGR)
                         pred_color=(0, 0, 255)):  # 红色 (BGR)

    # 1. 复制一份图像，以免污染原图
    combined_img = copy.deepcopy(img)
    # 2. 先画 GT (如果存在)
    if gt_bboxes is not None:
        combined_img = draw_lidar_bbox3d_on_img(
            gt_bboxes,
            combined_img,  # 在这张图上画
            proj_mat,
            img_metas=None,
            color=gt_color
        )
    # 3. 再画 Pred (如果存在)
    # 关键：传入的是已经画过 GT 的 combined_img
    if pred_bboxes is not None:
        combined_img = draw_lidar_bbox3d_on_img(
            pred_bboxes,
            combined_img,  # 接力画图
            proj_mat,
            img_metas=None,
            color=pred_color
        )
    # 4. 保存结果
    mmcv.mkdir_or_exist(out_dir)
    save_path = os.path.join(out_dir, f'{filename}.png')
    mmcv.imwrite(combined_img, save_path)


def inference(model, data_loader, device, weather=None):
    model.eval()
    with tqdm(total=len(data_loader)) as pbar:
        for i, (data, labels) in enumerate(data_loader):
            # Load data and labels (to device)
            labels: List[Dict[str, torch.Tensor]] = \
                [_dict_to(label, device) for label in labels]
            data: Dict[str, torch.Tensor] = \
                _dict_to(data, device)
            # Make prediction
            if model.__class__.__name__ != 'DPRT':
                output = model(data, labels, training=False, temperature=None)
            else:
                output = model(data)
            # class: 0背景 1轿车
            preds = decollate_batch(output)
            for pred, label in zip(preds, labels):
                sample_path = os.path.dirname(label['path'])
                seq = sample_path.split('/')[-2]
                sample_idx = sample_path.split('/')[-1]
                ids, kradar_labels = open_kradar_label_v1_1(seq, sample_idx)
                #############################################################
                # 不画有毫米波雷达看不见的框，不画有其他类别目标的框
                goon = True
                for kradar_label in kradar_labels:
                    if kradar_label['modality'] != 'R' or kradar_label['category'] != 'Sedan':
                        goon = False
                        break
                if goon == False:
                    continue
                #############################################################

                is_sedan_mask = torch.argmax(pred['class'] , dim=1) == 1
                sedan_indices = torch.nonzero(is_sedan_mask, as_tuple=False).reshape(-1)
                num_preds = sedan_indices.shape[0]  # 使用 shape[0] 而不是 len()
                if label['gt_center'].size(0) == 0 or label['gt_center'].size(0) != num_preds:
                    pbar.update(1)  # 即使跳过也要更新进度条
                    continue
                sedan_centers = pred['center'][sedan_indices] # 毫米波->激光 x+2.54 y-0.3 z-0.7
                sedan_size = pred['size'][sedan_indices]
                sedan_angles = pred['angle'][sedan_indices]
                sample_path = os.path.dirname(label['path'])
                img = read_img(os.path.join(sample_path, 'mono.jpg'))
                gt_bboxes_3d = convert_to_mmdet3d_format(label['gt_center'], label['gt_size'], label['gt_angle'])
                pred_bboxes_3d = convert_to_mmdet3d_format(sedan_centers, sedan_size, sedan_angles)
                lidar2img = np.load(os.path.join(sample_path, 'mono_info.npy'))
                show_combined_result(
                    img=img,
                    gt_bboxes=gt_bboxes_3d,  # 这里传 None
                    pred_bboxes=pred_bboxes_3d,  # 把你的框传给 pred_bboxes
                    proj_mat=lidar2img,  # 传入你的总投影矩阵
                    out_dir=f'/mnt/disk3/sushuang/bboxes_proj/{weather}',  # 图片会保存到当前目录
                    filename=f'{seq}_{sample_idx}_{ids["cam-front_idx"]}',
                )
                print(sample_path)
        return


def main(src: str, cfg: str, checkpoint: str, dst: str, split='test'):
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

    config['train']['batch_size'] = 2

    # Initialize test dataset
    test_normal_dataset = init(dataset=config['dataset'], src=src, split=split, config=config, weather_conditions=[0]) # 12164
    test_overcast_dataset = init(dataset=config['dataset'], src=src, split=split, config=config, weather_conditions=[1]) # 12164
    test_fog_dataset = init(dataset=config['dataset'], src=src, split=split, config=config, weather_conditions=[2]) # 12164
    test_rain_dataset = init(dataset=config['dataset'], src=src, split=split, config=config, weather_conditions=[3]) # 12164
    test_sleet_dataset = init(dataset=config['dataset'], src=src, split=split, config=config, weather_conditions=[4]) # 12164
    test_lightsnow_dataset = init(dataset=config['dataset'], src=src, split=split, config=config, weather_conditions=[5]) # 12164
    test_heavysnow_dataset = init(dataset=config['dataset'], src=src, split=split, config=config, weather_conditions=[6]) # 12164

    # config['train']['batch_size'] = 1

    # Load test dataset
    test_normal_loader = load(test_normal_dataset, config=config)
    test_overcast_loader = load(test_overcast_dataset, config=config)
    test_fog_loader = load(test_fog_dataset, config=config)
    test_rain_loader = load(test_rain_dataset, config=config)
    test_sleet_loader = load(test_sleet_dataset, config=config)
    test_lightsnow_loader = load(test_lightsnow_dataset, config=config)
    test_heavysnow_loader = load(test_heavysnow_dataset, config=config)

    # Evaluate model at checkpoint
    device = config['computing']['device']
    model, epoch, timestamp, optimizer_state_dict, scheduler_state_dict = load_model(checkpoint)
    model.to(device)
    # inference(model, test_normal_loader, device, 'normal')
    inference(model, test_overcast_loader, device, 'overcast')
    inference(model, test_fog_loader, device, 'fog')
    inference(model, test_rain_loader, device, 'rain')
    inference(model, test_sleet_loader, device, 'sleet')
    inference(model, test_lightsnow_loader, device, 'lightsnow')
    inference(model, test_heavysnow_loader, device, 'heavysnow')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT data preprocessing')
    parser.add_argument('--src', type=str, default='/mnt/disk2/sushuang/dpft_kradar/',
                        help="Path to the processed dataset folder.")
    parser.add_argument('--cfg', type=str,
                        default='/mnt/disk3/sushuang/checkpoints/20251204-100714-061-3/config.json',
                        # default='/mnt/disk3/sushuang/checkpoints/20251104-220208-232/config.json', # dpft
                        help="Path to the configuration file.")
    parser.add_argument('--checkpoint', type=str,
                        default='/mnt/disk3/sushuang/checkpoints/20251204-100714-061-3/checkpoints/20251204-100714-061-3_checkpoint_0023.pt',
                        # default='/mnt/disk3/sushuang/checkpoints/20251104-220208-232/checkpoints/20251104-220208-232_checkpoint_0045.pt', # dpft
                        help="Path to save the evaluation log.")
    parser.add_argument('--dst', type=str, default='/mnt/disk3/sushuang/evaluate_test',
                        help="Path to save the processed dataset.")
    args = parser.parse_args()
    print(args.src)
    print(args.cfg)
    print(args.checkpoint)
    epoch_str = args.checkpoint.split('/')[-1].split('.')[0].split('_')[-1]
    epoch = int(epoch_str)
    print(epoch)

    main(src=args.src, cfg=args.cfg, checkpoint=args.checkpoint, dst=args.dst)


