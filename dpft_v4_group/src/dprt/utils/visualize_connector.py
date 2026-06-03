"""
K-Radar 样本可视化接口
提供简洁的外部调用方式
"""

import os
from dprt.utils.visualize_all import vis_image_stream, vis_radar_ra_stream, vis_lidar_os1_stream,visualize_to_stream


def visualize(sample_dir: str, show: bool = True, save: bool = False):
    """
    可视化 K-Radar 样本的所有传感器数据（简化接口）.

    功能:
        - 可视化相机图像 (mono.jpg)
        - 可视化雷达 RCS 热力图 (ra.npy)
        - 可视化激光雷达 BEV 点云 (os1.npy, intensity 着色, BEV 过滤)

    Args:
        sample_dir: 样本目录路径
                   例如: "/path/to/kradar_processed2/test/12/00725_00689"
        show: 是否显示图像窗口 (默认: True)
        save: 是否保存到 sample_dir/visualization/ 目录 (默认: False)

    Returns:
        None

    Examples:
        >>> # 基本用法：只显示，不保存
        >>> visualize("/path/to/kradar_processed2/test/12/00725_00689")

        >>> # 显示并保存
        >>> visualize("/path/to/sample", show=True, save=True)

        >>> # 只保存不显示（用于批量处理）
        >>> visualize("/path/to/sample", show=False, save=True)
    """
    # 检查样本目录是否存在
    streams = visualize_to_stream(sample_dir, verbose=True)


# ============ 使用示例 ============
if __name__ == "__main__":
    # 示例 1: 只显示，不保存
    sample_dir = "/mnt/disk3/xiaxue/dataset/k_radar_v1/test/7/00183_00150"
    visualize(sample_dir)

    # 示例 2: 显示并保存
    # visualize(sample_dir, show=True, save=True)

    # 示例 3: 只保存不显示（批量处理）
    # visualize(sample_dir, show=False, save=True)