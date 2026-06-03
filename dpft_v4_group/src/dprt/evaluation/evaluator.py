from __future__ import annotations  # noqa: F407

from typing import Any, Callable, Dict, Iterable, List

import os.path as osp

import torch

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dprt.models import load as load_model
from dprt.evaluation.exporters import build as build_exporter
from dprt.evaluation.metric import build_metric


class CentralizedEvaluator():
    def __init__(self,
                 metric: torch.nn.modules.loss._Loss = None,
                 exporter: Callable = None,
                 device: str = None,
                 logging: str = None,):
        """
        Arguments:
            logging: Logging frequency. One of either None,
                step or epoch.
        """
        # Initialize instance arrtibutes
        self.eval_fn = metric
        self.export_fn = exporter
        self.device = device
        self.logging = logging

    @classmethod
    def from_config(cls,
                    config: Dict[str, Any],
                    *args,
                    **kwargs) -> CentralizedEvaluator:  # noqa: F821
        metric = build_metric(
            config['evaluate'],
            categories=config.get('data', {}).get('categories')
        )
        exporter = build_exporter(config['evaluate']['exporter']['name'], config)
        device = torch.device(config['computing']['device'])
        logging = config['train'].get('logging')

        return cls(
            metric=metric,
            exporter=exporter,
            device=device,
            logging=logging
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.evaluate(*args, **kwargs)

    @staticmethod
    def _dict_to(data: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in data.items()}

    @staticmethod
    def log_scalars(writer, scalars: Dict[str, Any], epoch: int, prefix: str = None) -> None:
        # Get prefix
        prefix = f"{prefix}/" if prefix is not None else ""

        # Add scalar values
        for name, scalar in scalars.items():
            writer.add_scalar(prefix + name, scalar, epoch)

    @torch.no_grad()
    def evaluate_complexity(self, epoch: int, model: torch.nn.Module,
                            data_loader: Iterable, writer=None):
        # Set model to evaluation mode
        model.eval()

        # Get inference test input
        data, _ = next(iter(data_loader))

        # Load test data (to device)
        data: Dict[str, torch.Tensor] = self._dict_to(data, self.device)

        # Determine model complexity
        with get_accelerator().device(self.device):
            flops, macs, params = get_model_profile(
                model=model, args=(data,),
                print_profile=False, warm_up=10, as_string=False
            )

        # Log model complexity
        self.log_scalars(
            writer, {'FLOPS': flops, 'MACS': macs, 'Parameters': params},
            epoch, 'test'
        )

    @torch.no_grad()
    def evaluate_inference_time(self, epoch: int, model: torch.nn.Module,
                                data_loader: Iterable, writer=None):
        # Set model to evaluation mode
        model.eval()

        # Get inference test input
        data, _ = next(iter(data_loader))

        # Load test data (to device)
        data: Dict[str, torch.Tensor] = self._dict_to(data, self.device)

        # Initialize loggers
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = torch.zeros((repetitions, 1))

        # GPU warm-up
        for _ in range(10):
            model(data)

        # Measure performance
        for rep in range(repetitions):
            starter.record()
            model(data)
            ender.record()
            # Wait for GPU sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

        # Calculate mean and std inference time in milliseconds
        mean_syn = torch.sum(timings) / repetitions
        std_syn = torch.std(timings)

        # Log inference time measures
        self.log_scalars(
            writer, {'Inference_time_mean_ms': mean_syn, 'Inference_time_std_ms': std_syn},
            epoch, 'test'
        )

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch: int, model: torch.nn.Module,
                           data_loader: Iterable, writer=None, dst: str = None):
        # Set model to evaluation mode
        model.eval()

        # Initialize epoch logs
        scalars = {}

        with tqdm(total=len(data_loader)) as pbar:
            for i, (data, labels) in enumerate(data_loader):
                # Load data and labels (to device)
                labels: List[Dict[str, torch.Tensor]] = \
                    [self._dict_to(label, self.device) for label in labels]
                data: Dict[str, torch.Tensor] = \
                    self._dict_to(data, self.device)

                # Make prediction
                output = model(data)

                # Evaluate model output
                metrics = self.eval_fn(output, labels)

                # Log evaluation step
                if self.logging == 'step':
                    self.log_scalars(writer, metrics, i + epoch * len(data_loader), 'test')

                # Add values to epoch log
                if self.logging == 'epoch':
                    for k, v in metrics.items():
                        scalars[k] = scalars.get(k, 0) + v

                # Export predictions
                if self.export_fn is not None:
                    self.export_fn(output, labels, i * len(labels), dst)

                # Report training progress
                pbar.update()

        if self.logging == 'epoch':
            # Average epoch logs
            scalars = {k: v / (i + 1) for k, v in scalars.items()}

            # Write epoch logs
            self.log_scalars(writer, scalars, epoch, 'test')

    def evaluate(self, checkpoint: str, data_loader: Iterable, dst: str = None):
        # Load model from checkpoint
        model, epoch, timestamp = load_model(checkpoint)

        # Load model (to device)
        model.to(self.device)

        # Check if destination is provided
        if self.logging is not None:
            dst = osp.join(dst, timestamp)

        # Initialize tensorboard writer (logging)
        if self.logging is not None:
            writer = SummaryWriter(log_dir=dst)

        # Evaluate model performance
        self.evaluate_one_epoch(epoch, model, data_loader, writer, dst)

        # Evaluate model inference time
        self.evaluate_inference_time(epoch, model, data_loader, writer)

        # Evaluate model complexity
        self.evaluate_complexity(epoch, model, data_loader, writer)

        # Flush and close writer
        if self.logging is not None:
            writer.flush()
            writer.close()


def build_evaluator(*args, **kwargs):
    return CentralizedEvaluator.from_config(*args, **kwargs)



# 可视化
# from __future__ import annotations
# from typing import Any, Callable, Dict, Iterable, List
# import os
# import os.path as osp
# import torch
# from deepspeed.profiling.flops_profiler import get_model_profile
# from deepspeed.accelerator import get_accelerator
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# from dprt.models import load as load_model
# from dprt.evaluation.exporters import build as build_exporter
# from dprt.evaluation.metric import build_metric
#
# # ==================== 新增导入 ====================
# import numpy as np
# import pandas as pd
# import cv2
#
# from dprt.training.assigner import SimpleHungarianAssigner
# from dprt.utils.project_bboxes import plot_3d_bbox
# from dprt.utils.calib import get_matrices_from_dict_calib
#
#
# class CentralizedEvaluator():
#
#     def __init__(self,
#                  metric: torch.nn.modules.loss._Loss = None,
#                  exporter: Callable = None,
#                  device: str = None,
#                  logging: str = None,
#                  enable_visualization: bool = False):  # 新增参数
#         """
#         Arguments:
#             logging: Logging frequency. One of either None, step or epoch.
#             enable_visualization: Whether to save prediction visualizations.
#         """
#         self.eval_fn = metric
#         self.export_fn = exporter
#         self.device = device
#         self.logging = logging
#         self.enable_visualization = enable_visualization  # 新增
#         self.vis_dir = None  # 新增
#
#     @classmethod
#     def from_config(cls, config: Dict[str, Any], *args, **kwargs) -> 'CentralizedEvaluator':
#         metric = build_metric(config['evaluate'])
#         exporter = build_exporter(config['evaluate']['exporter']['name'], config)
#         device = torch.device(config['computing']['device'])
#         logging = config['train'].get('logging')
#
#         # 新增：从配置读取可视化选项
#         enable_visualization = config['evaluate'].get('enable_visualization', False)
#
#         return cls(
#             metric=metric,
#             exporter=exporter,
#             device=device,
#             logging=logging,
#             enable_visualization=enable_visualization
#         )
#
#     def __call__(self, *args: Any, **kwargs: Any) -> Any:
#         self.evaluate(*args, **kwargs)
#
#     @staticmethod
#     def _dict_to(data: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
#         return {k: v.to(device) for k, v in data.items()}
#
#     @staticmethod
#     def log_scalars(writer, scalars: Dict[str, Any], epoch: int, prefix: str = None) -> None:
#         prefix = f"{prefix}/" if prefix is not None else ""
#         for name, scalar in scalars.items():
#             writer.add_scalar(prefix + name, scalar, epoch)
#
#     # ==================== 新增可视化方法 ====================
#     @staticmethod
#     def boxes_to_dataframe(boxes_dict: Dict[str, torch.Tensor]) -> pd.DataFrame:
#         """
#         将模型输出/标签转换为 DataFrame
#         """
#         center = boxes_dict['center'].cpu().numpy() if isinstance(boxes_dict['center'], torch.Tensor) else boxes_dict[
#             'center']
#         size = boxes_dict['size'].cpu().numpy() if isinstance(boxes_dict['size'], torch.Tensor) else boxes_dict['size']
#         angle = boxes_dict['angle'].cpu().numpy() if isinstance(boxes_dict['angle'], torch.Tensor) else boxes_dict[
#             'angle']
#
#         yaw = np.arctan2(angle[:, 0], angle[:, 1])
#         yaw_deg = np.rad2deg(yaw)
#
#         return pd.DataFrame({
#             'x': center[:, 0],
#             'y': center[:, 1],
#             'z': center[:, 2],
#             'yaw': yaw_deg,
#             'half_length': size[:, 0] / 2,
#             'half_width': size[:, 1] / 2,
#             'half_height': size[:, 2] / 2
#         })
#
#     @staticmethod
#     def labels_to_dataframe(labels: Dict[str, torch.Tensor]) -> pd.DataFrame:
#         """
#         将标签转换为 DataFrame
#         """
#         gt_center = labels['gt_center'].cpu().numpy() if isinstance(labels['gt_center'], torch.Tensor) else labels[
#             'gt_center']
#         gt_size = labels['gt_size'].cpu().numpy() if isinstance(labels['gt_size'], torch.Tensor) else labels['gt_size']
#         gt_angle = labels['gt_angle'].cpu().numpy() if isinstance(labels['gt_angle'], torch.Tensor) else labels[
#             'gt_angle']
#
#         yaw = np.arctan2(gt_angle[:, 0], gt_angle[:, 1])
#         yaw_deg = np.rad2deg(yaw)
#
#         return pd.DataFrame({
#             'x': gt_center[:, 0],
#             'y': gt_center[:, 1],
#             'z': gt_center[:, 2],
#             'yaw': yaw_deg,
#             'half_length': gt_size[:, 0] / 2,
#             'half_width': gt_size[:, 1] / 2,
#             'half_height': gt_size[:, 2] / 2
#         })
#
#     def visualize_predictions(self, data: Dict, output: Dict, labels: List[Dict],
#                               index_i: torch.Tensor, index_j: torch.Tensor, step: int):
#         """可视化匹配的预测框和真实框"""
#         try:
#             # 提取相机图像
#             camera_img = data.get('camera_mono')
#             if camera_img is None:
#                 return
#
#             # 转换为 numpy
#             if isinstance(camera_img, torch.Tensor):
#                 img = camera_img[0].cpu().numpy()
#                 if img.shape[0] in [1, 3]:
#                     img = np.transpose(img, (1, 2, 0))
#                 img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
#             else:
#                 return
#
#             # 获取标定参数
#             list_params = get_matrices_from_dict_calib()
#
#             # 提取匹配的预测框
#             batch_idx = 0
#             if len(index_i) > 0:
#                 matched_pred = {
#                     'center': output['center'][batch_idx][index_i],
#                     'size': output['size'][batch_idx][index_i],
#                     'angle': output['angle'][batch_idx][index_i],
#                 }
#                 pred_df = self.boxes_to_dataframe(matched_pred)
#             else:
#                 pred_df = pd.DataFrame()
#
#             # 提取匹配的真实框
#             if len(labels) > 0 and len(index_j) > 0:
#                 matched_gt = {
#                     'gt_center': labels[0]['gt_center'][index_j],
#                     'gt_size': labels[0]['gt_size'][index_j],
#                     'gt_angle': labels[0]['gt_angle'][index_j],
#                 }
#                 gt_df = self.labels_to_dataframe(matched_gt)
#             else:
#                 gt_df = pd.DataFrame()
#
#
#
#             # 取投影矩阵
#             P = data['label_to_camera_mono_p']
#             if isinstance(P, torch.Tensor):
#                 P = P[0].cpu().numpy()
#             # 绘制（现在可以指定颜色了）
#             img_with_boxes = img.copy()
#             # 先画绿色的 GT
#             if len(gt_df) > 0:
#                 img_with_boxes = plot_3d_bbox(img_with_boxes, gt_df, P, color=(0,255,0))
#
#             # 再画红色的预测
#             if len(pred_df) > 0:
#                 img_with_boxes = plot_3d_bbox(img_with_boxes, pred_df, P, color=(0, 0, 255))
#
#             # 添加文字
#             cv2.putText(img_with_boxes, f'Sample {step}', (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#             cv2.putText(img_with_boxes,
#                         f'Green: GT ({len(gt_df)}) | Red: Pred ({len(pred_df)}) | Matched: {len(index_i)}',
#                         (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#             # 保存
#             save_path = osp.join(self.vis_dir, f'prediction_{step:04d}.png')
#             cv2.imwrite(save_path, img_with_boxes)
#             print(f"  ✓ Saved: {save_path}")
#
#         except Exception as e:
#             print(f"  ⚠️  Visualization failed: {e}")
#             import traceback
#             traceback.print_exc()
#
#     # ==================== 其他评估方法保持不变 ====================
#     @torch.no_grad()
#     def evaluate_complexity(self, epoch: int, model: torch.nn.Module,
#                             data_loader: Iterable, writer=None):
#         model.eval()
#         data, _ = next(iter(data_loader))
#         data: Dict[str, torch.Tensor] = self._dict_to(data, self.device)
#
#         with get_accelerator().device(self.device):
#             flops, macs, params = get_model_profile(
#                 model=model, args=(data,),
#                 print_profile=False, warm_up=10, as_string=False
#             )
#
#         self.log_scalars(
#             writer, {'FLOPS': flops, 'MACS': macs, 'Parameters': params},
#             epoch, 'test'
#         )
#
#     @torch.no_grad()
#     def evaluate_inference_time(self, epoch: int, model: torch.nn.Module,
#                                 data_loader: Iterable, writer=None):
#         model.eval()
#         data, _ = next(iter(data_loader))
#         data: Dict[str, torch.Tensor] = self._dict_to(data, self.device)
#
#         starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#         repetitions = 300
#         timings = torch.zeros((repetitions, 1))
#
#         for _ in range(10):
#             model(data)
#
#         for rep in range(repetitions):
#             starter.record()
#             model(data)
#             ender.record()
#             torch.cuda.synchronize()
#             curr_time = starter.elapsed_time(ender)
#             timings[rep] = curr_time
#
#         mean_syn = torch.sum(timings) / repetitions
#         std_syn = torch.std(timings)
#
#         self.log_scalars(
#             writer, {'Inference_time_mean_ms': mean_syn, 'Inference_time_std_ms': std_syn},
#             epoch, 'test'
#         )
#
#     @torch.no_grad()
#     def evaluate_one_epoch(self, epoch: int, model: torch.nn.Module,
#                            data_loader: Iterable, writer=None, dst: str = None):
#         # Set model to evaluation mode
#         model.eval()
#
#         # Initialize epoch logs
#         scalars = {}
#
#         with tqdm(total=len(data_loader)) as pbar:
#             for i, (data, labels) in enumerate(data_loader):
#                 # Load data and labels (to device)
#                 labels: List[Dict[str, torch.Tensor]] = \
#                     [self._dict_to(label, self.device) for label in labels]
#                 data: Dict[str, torch.Tensor] = \
#                     self._dict_to(data, self.device)
#
#                 # Make prediction
#                 output = model(data)
#
#                 batch_size = output["center"].shape[0]
#
#                  # 将输出转换为列表结构而不是字典结构
#                 outputs_list = []
#                 for j in range(batch_size):
#                     outputs_list.append({
#                     "center": output["center"][j],
#                     "size": output["size"][j],
#                     "class": output["class"][j],
#                     "angle": output["angle"][j]
#                     })
#
#                 assigner = SimpleHungarianAssigner()
#                 index_i, index_j = assigner(outputs_list, labels)
#                 # ========== 修改：传入匹配索引 ==========
#                 if self.enable_visualization and self.vis_dir is not None:
#                     self.visualize_predictions(data, output, labels, index_i, index_j, i)
#
#                 # Evaluate model output
#                 metrics = self.eval_fn(output, labels)
#
#                 # Log evaluation step
#                 if self.logging == 'step':
#                     self.log_scalars(writer, metrics, i + epoch * len(data_loader), 'test')
#
#                 # Add values to epoch log
#                 if self.logging == 'epoch':
#                     for k, v in metrics.items():
#                         scalars[k] = scalars.get(k, 0) + v
#
#                 # Export predictions
#                 if self.export_fn is not None:
#                     self.export_fn(output, labels, i * len(labels), dst)
#
#                 # Report training progress
#                 pbar.update()
#
#         if self.logging == 'epoch':
#             # Average epoch logs
#             scalars = {k: v / (i + 1) for k, v in scalars.items()}
#             # Write epoch logs
#             self.log_scalars(writer, scalars, epoch, 'test')
#
#     def evaluate(self, checkpoint: str, data_loader: Iterable, dst: str = None):
#         # Load model from checkpoint
#         model, epoch, timestamp = load_model(checkpoint)
#
#         # Load model (to device)
#         model.to(self.device)
#
#         # Check if destination is provided
#         if self.logging is not None:
#             dst = osp.join(dst, timestamp)
#
#         # ========== 新增：创建可视化目录 ==========
#         if self.enable_visualization and dst is not None:
#             self.vis_dir = osp.join(dst, 'visualizations')
#             os.makedirs(self.vis_dir, exist_ok=True)
#             print(f"📁 Visualization output: {self.vis_dir}")
#
#         # Initialize tensorboard writer (logging)
#         if self.logging is not None:
#             writer = SummaryWriter(log_dir=dst)
#
#         # Evaluate model performance
#         self.evaluate_one_epoch(epoch, model, data_loader, writer, dst)
#
#         # Evaluate model inference time
#         self.evaluate_inference_time(epoch, model, data_loader, writer)
#
#         # Evaluate model complexity
#         self.evaluate_complexity(epoch, model, data_loader, writer)
#
#         # Flush and close writer
#         if self.logging is not None:
#             writer.flush()
#             writer.close()
#
#         # ========== 新增：打印总结 ==========
#         if self.enable_visualization:
#             print(f"\n🎉 Visualization completed! Saved to: {self.vis_dir}")
#
#
# def build_evaluator(*args, **kwargs):
#     return CentralizedEvaluator.from_config(*args, **kwargs)
