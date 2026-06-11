import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple, List, Dict
import io
import cv2
import pandas as pd
import torch
# ==================== 在文件开头添加这些导入 ====================
import copy
from dprt.utils.project_point_fusion import get_matrices_from_dict_calib
# import mmcv
# from mmdet3d.core.bbox import LiDARInstance3DBoxes
# from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

# ==================== 辅助函数：转换 Tensor 到 NumPy ====================

def to_numpy(tensor):
    """将 Tensor 转换为 NumPy 数组"""
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    return tensor


# ==================== 画框的核心函数 ====================

def draw_bev_box(ax, center, size, angle, color='r', linewidth=2, alpha=1.0):
    """
    在 BEV 视图上绘制单个 3D 框

    Args:
        ax: matplotlib axes 对象
        center: [x, y, z] 中心点（只用 x, y）
        size: [dx, dy, dz] 尺寸（只用 dx, dy）
        angle: [sin_yaw, cos_yaw] 或 yaw 角度
        color: 框颜色
        linewidth: 线条粗细
        alpha: 透明度
    """
    center = to_numpy(center)
    size = to_numpy(size)
    angle = to_numpy(angle)

    x, y = center[0], center[1]
    dx, dy = size[0], size[1]

    # 计算 yaw 角度
    if len(angle) == 2:
        yaw = -np.arctan2(angle[0], angle[1])
    else:
        yaw = angle

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # 计算 4 个角点（俯视图）
    corners_local = np.array([
        [-dx / 2, -dy / 2],
        [-dx / 2, dy / 2],
        [dx / 2, dy / 2],
        [dx / 2, -dy / 2],
        [-dx / 2, -dy / 2]  # 闭合
    ])

    # 旋转矩阵
    rot_mat = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

    # 旋转 + 平移
    corners_global = (rot_mat @ corners_local.T).T + np.array([x, y])

    # 在 BEV 图上绘制（注意坐标转换：-y vs x）
    ax.plot(-corners_global[:, 1], corners_global[:, 0],
            color=color, linewidth=linewidth, alpha=alpha)

    # 绘制朝向箭头
    arrow_length = dx / 3
    arrow_end_x = x + arrow_length * cos_yaw
    arrow_end_y = y + arrow_length * sin_yaw
    ax.arrow(-y, x, -(arrow_end_y - y), (arrow_end_x - x),
             head_width=0.3, head_length=0.2, fc=color, ec=color, alpha=alpha)


def cart_to_polar(x, y):
    """笛卡尔坐标转极坐标"""
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.degrees(np.arctan2(y, x))
    return r, theta


def draw_radar_box(ax, center, size, angle, color='r', linewidth=2, alpha=1.0,
                   coord_offset=(2.54, -0.3, -0.7)):
    """
    在 RA 图上绘制框的投影

    Args:
        ax: matplotlib axes 对象
        center: [x, y, z] 中心点
        size: [dx, dy, dz] 尺寸
        angle: [sin_yaw, cos_yaw] 或 yaw 角度
        color: 框颜色
        coord_offset: 雷达坐标偏移
    """
    center = to_numpy(center)
    size = to_numpy(size)
    angle = to_numpy(angle)

    # 应用坐标偏移
    x = center[0] + coord_offset[0]
    y = center[1] + coord_offset[1]
    dx, dy = size[0], size[1]

    # 计算 yaw 角度
    if len(angle) == 2:
        yaw = -np.arctan2(angle[0], angle[1])
    else:
        yaw = angle

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # 计算 4 个角点
    corners_local = np.array([
        [-dx / 2, -dy / 2],
        [-dx / 2, dy / 2],
        [dx / 2, dy / 2],
        [dx / 2, -dy / 2],
        [-dx / 2, -dy / 2]
    ])

    rot_mat = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

    corners = (rot_mat @ corners_local.T).T + np.array([x, y])

    # 转换为极坐标
    ranges = []
    azimuths = []
    for corner in corners:
        r, theta = cart_to_polar(corner[0], corner[1])
        ranges.append(r)
        azimuths.append(theta)

    # 绘制框
    ax.plot(azimuths, ranges, color=color, linewidth=linewidth, alpha=alpha)

    # 绘制中心点
    r_center, theta_center = cart_to_polar(x, y)
    ax.plot(theta_center, r_center, 'o', color=color, markersize=4, alpha=alpha)


def get_3d_box_corners(center, size, angle):
    """计算 3D 框的 8 个顶点坐标"""
    center = to_numpy(center)
    size = to_numpy(size)
    angle = to_numpy(angle)

    x, y, z = center[0], center[1], center[2]
    dx, dy, dz = size[0], size[1], size[2]

    if len(angle) == 2:
        yaw = -np.arctan2(angle[0], angle[1])
    else:
        yaw = angle

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    half_dx, half_dy, half_dz = dx / 2, dy / 2, dz / 2

    corners_obj = np.array([
        [-half_dx, -half_dy, -half_dz],
        [-half_dx, half_dy, -half_dz],
        [half_dx, half_dy, -half_dz],
        [half_dx, -half_dy, -half_dz],
        [-half_dx, -half_dy, half_dz],
        [-half_dx, half_dy, half_dz],
        [half_dx, half_dy, half_dz],
        [half_dx, -half_dy, half_dz],
    ])

    rot_mat = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])

    corners = corners_obj @ rot_mat.T + np.array([x, y, z])
    return corners


def project_3d_to_2d(corners_3d, proj_mat):
    """将 3D 点投影到 2D 图像平面"""
    num_points = corners_3d.shape[0]
    corners_3d_homo = np.hstack([corners_3d, np.ones((num_points, 1))])

    if proj_mat.shape[0] == 3:
        corners_2d_homo = corners_3d_homo @ proj_mat.T
    else:
        corners_2d_homo = corners_3d_homo @ proj_mat[:3, :].T

    depths = corners_2d_homo[:, 2]
    valid_mask = depths > 0

    corners_2d = corners_2d_homo[:, :2] / (corners_2d_homo[:, 2:3] + 1e-6)

    return corners_2d, valid_mask


# ==================== 新增：mmdet3d 格式转换函数 ====================
# def convert_to_mmdet3d_format(centers, sizes, angles, coord_offset=(8.54, -0.3, -0.7)):
#     """
#     将预测/GT框转换为 mmdet3d 的 LiDARInstance3DBoxes 对象
#     """
#     if len(centers) == 0:
#         return None
#
#     box_list = []
#     for i in range(len(centers)):
#         center = to_numpy(centers[i])
#         size = to_numpy(sizes[i])
#         angle = to_numpy(angles[i])
#
#         # 应用坐标偏移
#         box = [
#             center[0] + coord_offset[0],
#             center[1] + coord_offset[1],
#             center[2] + coord_offset[2],
#             size[0],  # dx
#             size[1],  # dy
#             size[2],  # dz
#             -np.arctan2(angle[0], angle[1])  # yaw
#         ]
#         box_list.append(box)
#
#     # 转换为 numpy array，再转 tensor
#     box_array = np.array(box_list, dtype=np.float32)
#     box_tensor = torch.from_numpy(box_array)
#
#     # 封装为 LiDARInstance3DBoxes
#     bboxes_3d = LiDARInstance3DBoxes(box_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))
#     return bboxes_3d
#
#
# def show_combined_result_mmdet3d(img, gt_bboxes, pred_bboxes, proj_mat,
#                                  gt_color=(0, 255, 0), pred_color=(0, 0, 255)):
#     """
#     使用 mmdet3d 的方法在图像上绘制 GT 和预测框
#     """
#     combined_img = copy.deepcopy(img)
#
#     # 绘制 GT（绿色）
#     # if gt_bboxes is not None:
#     #     combined_img = draw_lidar_bbox3d_on_img(
#     #         gt_bboxes,
#     #         combined_img,
#     #         proj_mat,
#     #         img_metas=None,
#     #         color=gt_color
#     #     )
#
#     # 绘制预测框（红色）
#     if pred_bboxes is not None:
#         combined_img = draw_lidar_bbox3d_on_img(
#             pred_bboxes,
#             combined_img,
#             proj_mat,
#             img_metas=None,
#             color=pred_color
#         )
#
#     return combined_img


def draw_3d_box_on_image(img, corners_2d, color=(0, 255, 0), thickness=2):
    """在图像上绘制 3D 框"""
    corners_2d = corners_2d.astype(np.int32)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for start, end in edges:
        pt1 = tuple(corners_2d[start])
        pt2 = tuple(corners_2d[end])
        cv2.line(img, pt1, pt2, color, thickness)

    return img


def draw_boxes_on_image(img, centers, sizes, angles, proj_mat, color=(0, 255, 0),
                        coord_offset=(2.54, -0.3, -0.7), thickness=2):
    """
    批量绘制多个 3D 框到图像上

    Args:
        img: 图像
        centers: (N, 3) 中心点坐标
        sizes: (N, 3) 尺寸
        angles: (N, 2) 角度 [sin, cos]
        proj_mat: 投影矩阵
        color: 框颜色
        coord_offset: 坐标系偏移（毫米波雷达 -> 激光雷达）
        thickness: 线条粗细

    Returns:
        img: 绘制后的图像
    """
    img_draw = img.copy()

    for i in range(len(centers)):
        # 坐标系转换
        center = centers[i].cpu().numpy() if torch.is_tensor(centers[i]) else centers[i]
        size = sizes[i].cpu().numpy() if torch.is_tensor(sizes[i]) else sizes[i]
        angle = angles[i].cpu().numpy() if torch.is_tensor(angles[i]) else angles[i]

        center[0] += coord_offset[0]
        center[1] += coord_offset[1]
        center[2] += coord_offset[2]

        # 计算 8 个顶点
        corners_3d = get_3d_box_corners(center, size, angle)

        # 投影到 2D
        corners_2d, valid_mask = project_3d_to_2d(corners_3d, proj_mat)

        # 如果至少有一个点在相机前方，就绘制
        if valid_mask.any():
            img_draw = draw_3d_box_on_image(img_draw, corners_2d, color, thickness)

    return img_draw



# ==================== 修改后的可视化函数（集成画框） ====================

# K-Radar 配置
BEV_X_RANGE = [0.0, 72.0]
BEV_Y_RANGE = [-50.0, 50.0]
BEV_Z_RANGE = [-2.0, 6.0]

AZIMUTH_RASTER = list(reversed([
    -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41,
    -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28,
    -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15,
    -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53
]))



# def vis_image_stream(
#         image_path: str,
#         proj_mat_path: Optional[str] = None,
#         pred_boxes: Optional[Dict] = None,
#         gt_boxes: Optional[Dict] = None,
#         coord_offset: Tuple[float, float, float] = (2.54, -0.3, -0.7),
#         figsize: Tuple[int, int] = (10, 6),
#         format: str = 'png',
#         dpi: int = 200
# ) -> Tuple[io.BytesIO, io.BytesIO]:
#     """
#     可视化相机图像，可选绘制 3D 框
#
#     Args:
#         image_path: 图像文件路径
#         proj_mat_path: 投影矩阵路径（如果需要画框）
#         pred_boxes: 预测框 {'centers': ..., 'sizes': ..., 'angles': ...}
#         gt_boxes: GT框 {'centers': ..., 'sizes': ..., 'angles': ...}
#         coord_offset: 坐标偏移
#
#     Returns:
#         (原始图像流, 带框图像流)
#     """
#     img = cv2.imread(image_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # 1. 保存原始图像
#     fig_orig, ax_orig = plt.subplots(figsize=figsize)
#     ax_orig.imshow(img_rgb)
#     ax_orig.axis("off")
#     plt.tight_layout(pad=0)
#
#     buffer_orig = io.BytesIO()
#     plt.savefig(buffer_orig, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
#     buffer_orig.seek(0)
#     plt.close(fig_orig)
#
#     # 2. 如果提供了框数据，绘制带框图像
#     if pred_boxes is not None or gt_boxes is not None:
#         if proj_mat_path is None:
#             raise ValueError("需要提供 proj_mat_path 才能绘制框")
#
#         proj_mat = np.load(proj_mat_path)
#         img_with_boxes = img.copy()
#
#         # 绘制 GT（绿色）
#         if gt_boxes is not None:
#             for i in range(len(gt_boxes['centers'])):
#                 center = to_numpy(gt_boxes['centers'][i]).copy()
#                 size = to_numpy(gt_boxes['sizes'][i])
#                 angle = to_numpy(gt_boxes['angles'][i])
#
#                 center[0] += coord_offset[0]
#                 center[1] += coord_offset[1]
#                 center[2] += coord_offset[2]
#
#                 corners_3d = get_3d_box_corners(center, size, angle)
#                 corners_2d, valid_mask = project_3d_to_2d(corners_3d, proj_mat)
#
#                 if valid_mask.any():
#                     img_with_boxes = draw_3d_box_on_image(img_with_boxes, corners_2d,
#                                                           color=(0, 255, 0), thickness=2)
#
#         # 绘制预测（红色）
#         if pred_boxes is not None:
#             for i in range(len(pred_boxes['centers'])):
#                 center = to_numpy(pred_boxes['centers'][i]).copy()
#                 size = to_numpy(pred_boxes['sizes'][i])
#                 angle = to_numpy(pred_boxes['angles'][i])
#
#                 center[0] += coord_offset[0]
#                 center[1] += coord_offset[1]
#                 center[2] += coord_offset[2]
#
#                 corners_3d = get_3d_box_corners(center, size, angle)
#                 corners_2d, valid_mask = project_3d_to_2d(corners_3d, proj_mat)
#
#                 if valid_mask.any():
#                     img_with_boxes = draw_3d_box_on_image(img_with_boxes, corners_2d,
#                                                           color=(0, 0, 255), thickness=2)
#
#         img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
#         fig_boxes, ax_boxes = plt.subplots(figsize=figsize)
#         ax_boxes.imshow(img_with_boxes_rgb)
#         ax_boxes.axis("off")
#         plt.tight_layout(pad=0)
#
#         buffer_boxes = io.BytesIO()
#         plt.savefig(buffer_boxes, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
#         buffer_boxes.seek(0)
#         plt.close(fig_boxes)
#     else:
#         buffer_boxes = None
#
#     return buffer_orig, buffer_boxes

# ==================== 修改后的mmdet相机可视化函数 ====================
# def vis_image_stream(
#         image_path: str,
#         proj_mat_path: Optional[str] = None,
#         pred_boxes: Optional[Dict] = None,
#         gt_boxes: Optional[Dict] = None,
#         coord_offset: Tuple[float, float, float] = (8.54, -0.3, -0.7),
#         figsize: Tuple[int, int] = (10, 6),
#         format: str = 'png',
#         dpi: int = 200
# ) -> Tuple[io.BytesIO, io.BytesIO]:
#     """
#     可视化相机图像，使用 mmdet3d 方法绘制 3D 框
#     """
#     img = cv2.imread(image_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # 1. 保存原始图像
#     fig_orig, ax_orig = plt.subplots(figsize=figsize)
#     ax_orig.imshow(img_rgb)
#     ax_orig.axis("off")
#     plt.tight_layout(pad=0)
#     buffer_orig = io.BytesIO()
#     plt.savefig(buffer_orig, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
#     buffer_orig.seek(0)
#     plt.close(fig_orig)
#
#     # 2. 如果提供了框数据，使用 mmdet3d 方法绘制
#     if pred_boxes is not None or gt_boxes is not None:
#         if proj_mat_path is None:
#             raise ValueError("需要提供 proj_mat_path 才能绘制框")
#
#         proj_mat = np.load(proj_mat_path)
#
#         # 转换为 mmdet3d 格式
#         gt_bboxes_3d = None
#         pred_bboxes_3d = None
#
#         if gt_boxes is not None and len(gt_boxes['centers']) > 0:
#             gt_bboxes_3d = convert_to_mmdet3d_format(
#                 gt_boxes['centers'],
#                 gt_boxes['sizes'],
#                 gt_boxes['angles'],
#                 coord_offset=coord_offset
#             )
#
#         if pred_boxes is not None and len(pred_boxes['centers']) > 0:
#             pred_bboxes_3d = convert_to_mmdet3d_format(
#                 pred_boxes['centers'],
#                 pred_boxes['sizes'],
#                 pred_boxes['angles'],
#                 coord_offset=coord_offset
#             )
#
#         # 使用 mmdet3d 方法绘制
#         img_with_boxes = show_combined_result_mmdet3d(
#             img=img,
#             gt_bboxes=gt_bboxes_3d,
#             pred_bboxes=pred_bboxes_3d,
#             proj_mat=proj_mat,
#             gt_color=(0, 255, 0),  # 绿色 GT
#             pred_color=(0, 0, 255)  # 红色预测
#         )
#
#         img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
#
#         fig_boxes, ax_boxes = plt.subplots(figsize=figsize)
#         ax_boxes.imshow(img_with_boxes_rgb)
#         ax_boxes.axis("off")
#         plt.tight_layout(pad=0)
#         buffer_boxes = io.BytesIO()
#         plt.savefig(buffer_boxes, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
#         buffer_boxes.seek(0)
#         plt.close(fig_boxes)
#     else:
#         buffer_boxes = None
#
#     return buffer_orig, buffer_boxes


# draw_boxes_on_image画图
# ==================== 恢复原始的相机可视化函数（不用 mmdet3d） ====================
def vis_image_stream(
        image_path: str,
        proj_mat_path: Optional[str] = None,
        pred_boxes: Optional[Dict] = None,
        gt_boxes: Optional[Dict] = None,
    coord_offset: Tuple[float, float, float] = (2.54, -0.3, -0.7),  # 与融合可视化保持一致
        figsize: Tuple[int, int] = (10, 6),
        format: str = 'png',
        dpi: int = 200
) -> Tuple[io.BytesIO, io.BytesIO]:
    """
    可视化相机图像，使用 draw_boxes_on_image 方法绘制 3D 框

    Args:
        image_path: 图像文件路径
        proj_mat_path: 投影矩阵路径（如果需要画框）
        pred_boxes: 预测框 {'centers': ..., 'sizes': ..., 'angles': ...}
        gt_boxes: GT框 {'centers': ..., 'sizes': ..., 'angles': ...}
        coord_offset: 坐标偏移

    Returns:
        (原始图像流, 带框图像流)
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. 保存原始图像
    fig_orig, ax_orig = plt.subplots(figsize=figsize)
    ax_orig.imshow(img_rgb)
    ax_orig.axis("off")
    plt.tight_layout(pad=0)

    buffer_orig = io.BytesIO()
    plt.savefig(buffer_orig, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
    buffer_orig.seek(0)
    plt.close(fig_orig)

    # 2. 如果提供了框数据，使用 draw_boxes_on_image 绘制带框图像
    if pred_boxes is not None or gt_boxes is not None:
        if proj_mat_path is None:
            _, intrinsics, _, T_ldr2cam = get_matrices_from_dict_calib()
            proj_mat = intrinsics @ T_ldr2cam[:3, :]
        else:
            proj_mat = np.load(proj_mat_path)
        img_with_boxes = img.copy()

        # 绘制 GT（绿色）
        if gt_boxes is not None and len(gt_boxes['centers']) > 0:
            img_with_boxes = draw_boxes_on_image(
                img_with_boxes,
                gt_boxes['centers'],
                gt_boxes['sizes'],
                gt_boxes['angles'],
                proj_mat,
                color=(0, 255, 0),  # 绿色
                coord_offset=coord_offset,
                thickness=2
            )

        # 绘制预测（红色）
        if pred_boxes is not None and len(pred_boxes['centers']) > 0:
            img_with_boxes = draw_boxes_on_image(
                img_with_boxes,
                pred_boxes['centers'],
                pred_boxes['sizes'],
                pred_boxes['angles'],
                proj_mat,
                color=(0, 0, 255),  # 红色
                coord_offset=coord_offset,
                thickness=2
            )

        img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

        fig_boxes, ax_boxes = plt.subplots(figsize=figsize)
        ax_boxes.imshow(img_with_boxes_rgb)
        ax_boxes.axis("off")
        plt.tight_layout(pad=0)

        buffer_boxes = io.BytesIO()
        plt.savefig(buffer_boxes, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
        buffer_boxes.seek(0)
        plt.close(fig_boxes)
    else:
        buffer_boxes = None

    return buffer_orig, buffer_boxes


def vis_lidar_os1_stream(
        os1_npy_path: str,
        pred_boxes: Optional[Dict] = None,
        gt_boxes: Optional[Dict] = None,
        color_by: str = "reflectivity",
        point_size: float = 0.5,
        max_points: Optional[int] = 200000,
        bev_filter: bool = True,
        x_range: Optional[List[float]] = None,
        y_range: Optional[List[float]] = None,
        z_range: Optional[List[float]] = None,
        figsize: Tuple[int, int] = (10, 8),
        format: str = 'png',
        dpi: int = 200,
        verbose: bool = False
) -> Tuple[io.BytesIO, io.BytesIO]:
    """
    可视化激光雷达 OS1-128 点云，可选绘制 BEV 框

    Args:
        pred_boxes: 预测框 {'centers': ..., 'sizes': ..., 'angles': ...}
        gt_boxes: GT框 {'centers': ..., 'sizes': ..., 'angles': ...}

    Returns:
        (原始点云图像流, 带框点云图像流)
    """
    if x_range is None:
        x_range = BEV_X_RANGE
    if y_range is None:
        y_range = BEV_Y_RANGE
    if z_range is None:
        z_range = BEV_Z_RANGE

    # 加载点云
    pts = np.load(os1_npy_path, allow_pickle=True)
    if pts.ndim > 2:
        pts = pts.reshape(-1, pts.shape[-1])

    xyz = pts[:, :3].astype(np.float32)

    # 过滤
    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]
    pts = pts[mask]

    if bev_filter:
        x_mask = (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] <= x_range[1])
        y_mask = (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] <= y_range[1])
        z_mask = (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])
        bev_mask = x_mask & y_mask & z_mask
        xyz = xyz[bev_mask]
        pts = pts[bev_mask]

    # 降采样
    if max_points and xyz.shape[0] > max_points:
        indices = np.random.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[indices]
        pts = pts[indices]

    # 颜色
    color_map = {
        "z": (xyz[:, 2], "Height (m)"),
        "intensity": (pts[:, 3], "Intensity"),
        "reflectivity": (pts[:, 5], "Reflectivity"),
        "range": (pts[:, 8], "Range (m)")
    }

    if color_by not in color_map:
        color_by = "reflectivity"
    color_data, _ = color_map[color_by]
    color_data = (color_data - color_data.min()) / (color_data.max() - color_data.min() + 1e-8)
    color_data = np.clip(color_data, 0, 1)

    # 1. 绘制原始点云
    fig_orig, ax_orig = plt.subplots(figsize=figsize)
    ax_orig.scatter(-xyz[:, 1], xyz[:, 0], s=point_size, c=color_data,
                    cmap="viridis", alpha=0.8)
    ax_orig.set_aspect("equal", "box")
    ax_orig.set_xlim(-y_range[1], -y_range[0])
    ax_orig.set_ylim(x_range[0], x_range[1])
    ax_orig.axis("off")
    plt.tight_layout(pad=0)

    buffer_orig = io.BytesIO()
    plt.savefig(buffer_orig, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
    buffer_orig.seek(0)
    plt.close(fig_orig)

    # 2. 如果提供了框数据，绘制带框图像
    if pred_boxes is not None or gt_boxes is not None:
        fig_boxes, ax_boxes = plt.subplots(figsize=figsize)
        ax_boxes.scatter(-xyz[:, 1], xyz[:, 0], s=point_size, c=color_data,
                         cmap="viridis", alpha=0.8)

        # 绘制 GT（绿色）
        if gt_boxes is not None and len(gt_boxes['centers']) > 0:
            for i in range(len(gt_boxes['centers'])):
                draw_bev_box(ax_boxes, gt_boxes['centers'][i], gt_boxes['sizes'][i],
                             gt_boxes['angles'][i], color='green', linewidth=2)

        # 绘制预测（红色）
        if pred_boxes is not None and len(pred_boxes['centers']) > 0:
            for i in range(len(pred_boxes['centers'])):
                draw_bev_box(ax_boxes, pred_boxes['centers'][i], pred_boxes['sizes'][i],
                             pred_boxes['angles'][i], color='red', linewidth=2)

        ax_boxes.set_aspect("equal", "box")
        ax_boxes.set_xlim(-y_range[1], -y_range[0])
        ax_boxes.set_ylim(x_range[0], x_range[1])
        ax_boxes.axis("off")
        plt.tight_layout(pad=0)

        buffer_boxes = io.BytesIO()
        plt.savefig(buffer_boxes, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
        buffer_boxes.seek(0)
        plt.close(fig_boxes)
    else:
        buffer_boxes = None

    return buffer_orig, buffer_boxes

#
# def vis_radar_ra_stream(
#         ra_npy_path: str,
#         pred_boxes: Optional[Dict] = None,
#         gt_boxes: Optional[Dict] = None,
#         channel: int = 0,
#         log_scale: bool = True,
#         figsize: Tuple[int, int] = (10, 6),
#         format: str = 'png',
#         dpi: int = 200
# ) -> Tuple[io.BytesIO, io.BytesIO]:
#     """
#     可视化毫米波雷达 RA 热力图，可选绘制框
#
#     Args:
#         pred_boxes: 预测框 {'centers': ..., 'sizes': ..., 'angles': ...}
#         gt_boxes: GT框 {'centers': ..., 'sizes': ..., 'angles': ...}
#
#     Returns:
#         (原始RA图像流, 带框RA图像流)
#     """
#     ra = np.load(ra_npy_path, allow_pickle=True)
#
#     if ra.ndim == 3 and ra.shape[2] == 6:
#         ra_vis = ra[:, :, channel]
#     else:
#         ra_vis = np.abs(ra)
#
#     if channel < 3 and log_scale:
#         ra_vis = 20 * np.log10(np.maximum(ra_vis, 1e-10))
#     else:
#         ra_vis = np.abs(ra_vis)
#
#     # 1. 绘制原始RA图
#     fig_orig, ax_orig = plt.subplots(figsize=figsize)
#     ax_orig.imshow(ra_vis, origin="upper", aspect="auto", cmap="jet")
#     ax_orig.axis("off")
#     plt.tight_layout(pad=0)
#
#     buffer_orig = io.BytesIO()
#     plt.savefig(buffer_orig, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
#     buffer_orig.seek(0)
#     plt.close(fig_orig)
#
#     # 2. 如果提供了框数据，绘制带框图像
#     if pred_boxes is not None or gt_boxes is not None:
#         from matplotlib.collections import LineCollection
#
#         fig_boxes, ax_boxes = plt.subplots(figsize=figsize)
#         ax_boxes.imshow(ra_vis, origin="upper", aspect="auto", cmap="jet", alpha=0.7,
#                         extent=[AZIMUTH_RASTER[-1], AZIMUTH_RASTER[0], 0, 118])
#
#         # 绘制 GT（绿色）
#         # if gt_boxes is not None and len(gt_boxes['centers']) > 0:
#         #     for i in range(len(gt_boxes['centers'])):
#         #         draw_radar_box(ax_boxes, gt_boxes['centers'][i], gt_boxes['sizes'][i],
#         #                        gt_boxes['angles'][i], color='green', linewidth=2)
#
#         # 绘制预测（红色）
#         if pred_boxes is not None and len(pred_boxes['centers']) > 0:
#             for i in range(len(pred_boxes['centers'])):
#                 draw_radar_box(ax_boxes, pred_boxes['centers'][i], pred_boxes['sizes'][i],
#                                pred_boxes['angles'][i], color='red', linewidth=2)
#
#         # ax_boxes.set_xlabel('Azimuth (°)', fontsize=10)
#         # ax_boxes.set_ylabel('Range (m)', fontsize=10)
#         # ax_boxes.set_xlim(AZIMUTH_RASTER[-1], AZIMUTH_RASTER[0])
#         # ax_boxes.set_ylim(0, 118)
#         plt.tight_layout(pad=0)
#
#         buffer_boxes = io.BytesIO()
#         plt.savefig(buffer_boxes, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
#         buffer_boxes.seek(0)
#         plt.close(fig_boxes)
#     else:
#         buffer_boxes = None
#
#     return buffer_orig, buffer_boxes

# 【新增】从visualize_fusion导入RA可视化函数
# 【新增】从visualize_fusion导入RA可视化函数
# ==================== 【替换】基于show_kradar_ra.py的RA可视化函数 ====================

def visradarrastream_clean(ranpypath: str,
                           channel: int = 0,  # 添加channel参数
                           figsize: Tuple[int, int] = (8, 12),  # 调整宽高比
                           format: str = 'png',
                           dpi: int = 200) -> io.BytesIO:  # 降低DPI避免过大
    """
    K-Radar RA可视化 - 支持(256, 107, 6)格式
    基于visualize_fusion.py的配置
    """
    # 加载数据
    ra = np.load(ranpypath, allow_pickle=True)

    print(f"[调试] RA数据形状: {ra.shape}")

    # 处理(256, 107, 6)格式
    if ra.ndim == 3 and ra.shape[2] == 6:
        # 选择指定通道（默认channel 0: RCS max）
        ra_vis = ra[:, :, channel]
        print(f"[调试] 使用通道 {channel}")
    elif ra.ndim == 3:
        # 其他多通道格式：平均所有通道
        radar_power = np.abs(ra) ** 2
        ra_vis = np.mean(radar_power, axis=2)
        print(f"[调试] 平均 {ra.shape[2]} 个通道")
    else:
        # 单通道数据
        ra_vis = np.abs(ra)

    print(f"[调试] 处理后形状: {ra_vis.shape}")
    print(f"[调试] 数据范围: [{ra_vis.min():.2f}, {ra_vis.max():.2f}]")

    # 对RCS通道使用对数刻度（channel 0-2）
    if channel < 3:
        ra_vis = 20 * np.log10(np.maximum(ra_vis, 1e-10))
        print(f"[调试] 应用对数刻度: [{ra_vis.min():.2f}, {ra_vis.max():.2f}] dB")
    else:
        # Doppler通道取绝对值
        ra_vis = np.abs(ra_vis)

    # 创建画布
    fig, ax = plt.subplots(figsize=figsize)

    # K-Radar标准配置
    AZIMUTH_MIN = -53  # 度
    AZIMUTH_MAX = 53
    RANGE_MAX = 118.03  # 米（最后一个距离bin）

    # imshow可视化
    im = ax.imshow(ra_vis,
                   origin='upper',  # K-Radar使用upper
                   aspect='auto',
                   extent=[AZIMUTH_MIN, AZIMUTH_MAX, RANGE_MAX, 0],  # [左,右,下,上]
                   cmap='jet',
                   interpolation='bilinear')  # 平滑插值

    # 移除装饰
    ax.axis('off')

    # 保存到BytesIO
    buffer = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    plt.close(fig)

    return buffer


# ==================== 【修改】更新vis_radar_ra_stream函数 ====================
# ==================== 更新vis_radar_ra_stream ====================
def vis_radar_ra_stream(
        ra_npy_path: str,
        pred_boxes: Optional[Dict] = None,
        gt_boxes: Optional[Dict] = None,
        channel: int = 0,
        log_scale: bool = True,  # 保留参数但不再使用（已在内部处理）
        figsize: Tuple[int, int] = (8, 12),
        format: str = 'png',
        dpi: int = 200,
        use_clean_version: bool = True
) -> Tuple[io.BytesIO, io.BytesIO]:
    """
    可视化K-Radar RA热力图 - 支持(256, 107, 6)格式
    """
    # 1. 生成原始图像
    if use_clean_version:
        buffer_orig = visradarrastream_clean(
            ranpypath=ra_npy_path,
            channel=channel,
            figsize=figsize,
            format=format,
            dpi=dpi
        )
    else:
        buffer_orig = visradarrastream_with_axes(
            ranpypath=ra_npy_path,
            channel=channel,
            figsize=figsize,
            format=format,
            dpi=dpi
        )

    # 2. 绘制带框图像（如果提供）
    if pred_boxes is not None or gt_boxes is not None:
        ra = np.load(ra_npy_path, allow_pickle=True)

        # 处理数据
        if ra.ndim == 3 and ra.shape[2] == 6:
            ra_vis = ra[:, :, channel]
        else:
            ra_vis = np.abs(ra)

        # 对数刻度
        if channel < 3:
            ra_vis = 20 * np.log10(np.maximum(ra_vis, 1e-10))

        # 绘制
        fig_boxes, ax_boxes = plt.subplots(figsize=figsize)
        ax_boxes.imshow(ra_vis,
                        origin='upper',
                        aspect='auto',
                        extent=[-53, 53, 118.03, 0],
                        cmap='jet',
                        interpolation='bilinear',
                        alpha=0.7)

        # 绘制 GT（绿色）
        if gt_boxes is not None and len(gt_boxes['centers']) > 0:
            for i in range(len(gt_boxes['centers'])):
                draw_radar_box(ax_boxes, gt_boxes['centers'][i],
                               gt_boxes['sizes'][i],
                               gt_boxes['angles'][i],
                               color='green', linewidth=2)

        # 绘制预测框（红色）
        if pred_boxes is not None and len(pred_boxes['centers']) > 0:
            for i in range(len(pred_boxes['centers'])):
                draw_radar_box(ax_boxes, pred_boxes['centers'][i],
                               pred_boxes['sizes'][i],
                               pred_boxes['angles'][i],
                               color='red', linewidth=2)

        ax_boxes.axis('off')
        plt.tight_layout(pad=0)

        buffer_boxes = io.BytesIO()
        plt.savefig(buffer_boxes, format=format, dpi=dpi,
                    bbox_inches='tight', pad_inches=0)
        buffer_boxes.seek(0)
        plt.close(fig_boxes)
    else:
        buffer_boxes = None

    return buffer_orig, buffer_boxes
def visualize_with_predictions(
        sample_dir: str,
        predictions: Optional[Dict] = None,
        format: str = 'png',
        dpi: int = 200,
        verbose: bool = False
) -> Dict[str, Dict[str, io.BytesIO]]:
    """
    可视化样本并绘制预测框

    Args:
        sample_dir: 样本目录路径
        predictions: 预测结果字典
            {
                'camera': {'pred': {...}, 'gt': {...}},
                'lidar': {'pred': {...}, 'gt': {...}},
                'radar': {'pred': {...}, 'gt': {...}},
                'full': {'pred': {...}, 'gt': {...}}
            }
            其中 pred/gt 包含 'centers', 'sizes', 'angles'

    Returns:
        streams: 包含所有图像字节流的字典
            {
                'camera': {'original': BytesIO, 'with_boxes': BytesIO},
                'lidar': {'original': BytesIO, 'with_boxes': BytesIO},
                'radar': {'original': BytesIO, 'with_boxes': BytesIO}
            }
    """
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"样本目录不存在: {sample_dir}")

    mono_path = os.path.join(sample_dir, "mono.jpg")
    mono_info_path = os.path.join(sample_dir, "mono_info.npy")
    ra_path = os.path.join(sample_dir, "ra.npy")
    os1_path = os.path.join(sample_dir, "os1.npy")

    if verbose:
        print(f"可视化样本: {os.path.basename(sample_dir)}")

    streams = {}

    # 1. Camera
    if verbose:
        print("[1/3] 生成相机图像...")

    pred_boxes = None
    gt_boxes = None

    if predictions and 'camera' in predictions:
        pred_boxes = predictions['camera'].get('pred')
        gt_boxes = predictions['camera'].get('gt')

    orig, boxes = vis_image_stream(
        mono_path,
        proj_mat_path=None,
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        format=format,
        dpi=dpi
    )
    streams['camera'] = {'original': orig, 'with_boxes': boxes}

    # 2. Radar
    if verbose:
        print("[2/3] 生成雷达热力图...")

    pred_boxes = None
    gt_boxes = None

    if predictions and 'radar' in predictions:
        pred_boxes = predictions['radar'].get('pred')
        gt_boxes = predictions['radar'].get('gt')

        # 【关键】直接使用预处理数据集的ra.npy
    ra_path = os.path.join(sample_dir, "ra.npy")

    if not os.path.exists(ra_path):
        print(f"⚠️ 警告: ra.npy不存在: {ra_path}")
        streams['radar'] = {'original': None, 'with_boxes': None}
    else:
        print("BBBBBB可视化：：",ra_path)
        orig, boxes = vis_radar_ra_stream(
            ra_path,  # 使用预处理的ra.npy
            pred_boxes=pred_boxes,
            gt_boxes=gt_boxes,
            channel=0,
            format=format,
            dpi=dpi,
            use_clean_version=True
        )
    streams['radar'] = {'original': orig, 'with_boxes': boxes}

    # 3. LiDAR
    if verbose:
        print("[3/3] 生成激光雷达点云...")

    pred_boxes = None
    gt_boxes = None

    if predictions and 'lidar' in predictions:
        pred_boxes = predictions['lidar'].get('pred')
        gt_boxes = predictions['lidar'].get('gt')

    orig, boxes = vis_lidar_os1_stream(
        os1_path,
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        color_by="reflectivity",
        bev_filter=True,
        format=format,
        dpi=dpi,
        verbose=verbose
    )
    streams['lidar'] = {'original': orig, 'with_boxes': boxes}

    if verbose:
        print("✓ 完成")

    return streams


# ==================== 使用示例 ====================

if __name__ == "__main__":
    sample_dir = "/mnt/disk3/xiaxue/dataset/k_radar_v1/test/31/00156_00150"

    # 示例：不带预测框（原始可视化）
    print("\n=== 示例 1: 原始可视化（无框） ===")
    streams = visualize_with_predictions(sample_dir, predictions=None, verbose=True)

    save_dir = "/mnt/disk3/xiaxue/code/fusion_backend/results"
    os.makedirs(save_dir, exist_ok=True)

    for modality, stream_dict in streams.items():
        # 保存原始图像
        if stream_dict['original']:
            save_path = os.path.join(save_dir, f"{modality}_original.png")
            with open(save_path, "wb") as f:
                f.write(stream_dict['original'].getvalue())
            print(f"已保存: {save_path}")

    # 示例：带预测框（需要从 inference_mml.py 获取预测结果）
    # predictions = {
    #     'camera': {
    #         'pred': {'centers': ..., 'sizes': ..., 'angles': ...},
    #         'gt': {'centers': ..., 'sizes': ..., 'angles': ...}
    #     },
    #     'lidar': {...},
    #     'radar': {...}
    # }
    # streams = visualize_with_predictions(sample_dir, predictions=predictions, verbose=True)
