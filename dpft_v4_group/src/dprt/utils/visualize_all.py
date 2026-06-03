import argparse
from dprt.datasets import init
from dprt.datasets import load
from dprt.models import load as load_model
# from evaluator import SimpleEvaluator
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed
import os
import os.path as osp
import torch.utils.data as data
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple, List, Dict
import io

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# K-Radar BEV 范围配置（来自 kradar.json）
BEV_X_RANGE = [0.0, 20.0]
BEV_Y_RANGE = [-6.4, 6.4]
BEV_Z_RANGE = [-2.0, 6.0]

# K-Radar 雷达配置
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

RANGE_RASTER = [
    0.0, 0.46289062, 0.92578125, 1.38867188,
    1.8515625, 2.31445312, 2.77734375, 3.24023438,
    3.703125, 4.16601562, 4.62890625, 5.09179688,
    5.5546875, 6.01757812, 6.48046875, 6.94335938,
    7.40625, 7.86914062, 8.33203125, 8.79492188,
    9.2578125, 9.72070312, 10.18359375, 10.64648438,
    11.109375, 11.57226562, 12.03515625, 12.49804688,
    12.9609375, 13.42382812, 13.88671875, 14.34960938,
    14.8125, 15.27539062, 15.73828125, 16.20117188,
    16.6640625, 17.12695312, 17.58984375, 18.05273438,
    18.515625, 18.97851562, 19.44140625, 19.90429688,
    20.3671875, 20.83007812, 21.29296875, 21.75585938,
    22.21875, 22.68164062, 23.14453125, 23.60742188,
    24.0703125, 24.53320312, 24.99609375, 25.45898438,
    25.921875, 26.38476562, 26.84765625, 27.31054688,
    27.7734375, 28.23632812, 28.69921875, 29.16210938,
    29.625, 30.08789062, 30.55078125, 31.01367188,
    31.4765625, 31.93945312, 32.40234375, 32.86523438,
    33.328125, 33.79101562, 34.25390625, 34.71679688,
    35.1796875, 35.64257812, 36.10546875, 36.56835938,
    37.03125, 37.49414062, 37.95703125, 38.41992188,
    38.8828125, 39.34570312, 39.80859375, 40.27148438,
    40.734375, 41.19726562, 41.66015625, 42.12304688,
    42.5859375, 43.04882812, 43.51171875, 43.97460938,
    44.4375, 44.90039062, 45.36328125, 45.82617188,
    46.2890625, 46.75195312, 47.21484375, 47.67773438,
    48.140625, 48.60351562, 49.06640625, 49.52929688,
    49.9921875, 50.45507812, 50.91796875, 51.38085938,
    51.84375, 52.30664062, 52.76953125, 53.23242188,
    53.6953125, 54.15820312, 54.62109375, 55.08398438,
    55.546875, 56.00976562, 56.47265625, 56.93554688,
    57.3984375, 57.86132812, 58.32421875, 58.78710938,
    59.25, 59.71289062, 60.17578125, 60.63867188,
    61.1015625, 61.56445312, 62.02734375, 62.49023438,
    62.953125, 63.41601562, 63.87890625, 64.34179688,
    64.8046875, 65.26757812, 65.73046875, 66.19335938,
    66.65625, 67.11914062, 67.58203125, 68.04492188,
    68.5078125, 68.97070312, 69.43359375, 69.89648438,
    70.359375, 70.82226562, 71.28515625, 71.74804688,
    72.2109375, 72.67382812, 73.13671875, 73.59960938,
    74.0625, 74.52539062, 74.98828125, 75.45117188,
    75.9140625, 76.37695312, 76.83984375, 77.30273438,
    77.765625, 78.22851562, 78.69140625, 79.15429688,
    79.6171875, 80.08007812, 80.54296875, 81.00585938,
    81.46875, 81.93164062, 82.39453125, 82.85742188,
    83.3203125, 83.78320312, 84.24609375, 84.70898438,
    85.171875, 85.63476562, 86.09765625, 86.56054688,
    87.0234375, 87.48632812, 87.94921875, 88.41210938,
    88.875, 89.33789062, 89.80078125, 90.26367188,
    90.7265625, 91.18945312, 91.65234375, 92.11523438,
    92.578125, 93.04101562, 93.50390625, 93.96679688,
    94.4296875, 94.89257812, 95.35546875, 95.81835938,
    96.28125, 96.74414062, 97.20703125, 97.66992188,
    98.1328125, 98.59570312, 99.05859375, 99.52148438,
    99.984375, 100.44726562, 100.91015625, 101.37304688,
    101.8359375, 102.29882812, 102.76171875, 103.22460938,
    103.6875, 104.15039062, 104.61328125, 105.07617188,
    105.5390625, 106.00195312, 106.46484375, 106.92773438,
    107.390625, 107.85351562, 108.31640625, 108.77929688,
    109.2421875, 109.70507812, 110.16796875, 110.63085938,
    111.09375, 111.55664062, 112.01953125, 112.48242188,
    112.9453125, 113.40820312, 113.87109375, 114.33398438,
    114.796875, 115.25976562, 115.72265625, 116.18554688,
    116.6484375, 117.11132812, 117.57421875, 118.03710938
]

N_RANGE = len(RANGE_RASTER)
N_AZIMUTH = len(AZIMUTH_RASTER)
RANGE_MIN = RANGE_RASTER[0]
RANGE_MAX = RANGE_RASTER[-1]
AZIMUTH_MIN = AZIMUTH_RASTER[-1]
AZIMUTH_MAX = AZIMUTH_RASTER[0]

#可视化
def vis_image_stream(
        image_path: str,
        figsize: Tuple[int, int] = (10, 6),
        format: str = 'png',
        dpi: int = 200
)  -> io.BytesIO:
    """
    可视化相机图像（mono.jpg 或 stereo.jpg）.

    Args:
        image_path: 图像文件路径
        title: 图像标题（默认使用文件名）
        figsize: 图像尺寸
        save_path: 保存路径（可选）
        show: 是否显示图像

    Returns:
       buffer: 图像字节流 (BytesIO)
    """
    img = np.array(Image.open(image_path))

    fig=plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0)

    # if save_path:
    #     plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    # if show:
    #     plt.show()
    # else:
    #     plt.close()

    # 保存到内存缓冲区
    buffer = io.BytesIO()
    plt.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    plt.show()
    # plt.close(fig)

    return buffer


def vis_radar_ra_stream(
    ra_npy_path: str,
    channel: int = 0,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    format: str = 'png',
    dpi: int = 200
) -> io.BytesIO:
    """
    可视化毫米波雷达 RA（Range-Azimuth）热力图.

    K-Radar ra.npy 格式: (range_bins, azimuth_bins, 6)
    - Channel 0-2: RCS (max, median, var)
    - Channel 3-5: Doppler (max, median, var)

    Args:
        ra_npy_path: ra.npy 文件路径
        channel: 选择可视化的通道 (0-5)
        channel_names: 通道名称列表（默认使用标准命名）
        log_scale: 是否使用对数刻度（仅对 RCS 通道）
        figsize: 图像尺寸
        save_path: 保存路径（可选）
        show: 是否显示图像

    Returns:
        ra: buffer
    """
    ra = np.load(ra_npy_path, allow_pickle=True)

    # 提取指定通道
    if ra.ndim == 3 and ra.shape[2] == 6:
        ra_vis = ra[:, :, channel]
    else:
        # 兼容其他格式（取幅度）
        ra_vis = np.abs(ra)

    # 对 RCS 通道使用对数刻度
    if channel < 3:  # 只对 RCS 做对数化
        if log_scale:
            ra_vis = 20 * np.log10(np.maximum(ra_vis, 1e-10))
    else:  # Doppler 取绝对值
        ra_vis = np.abs(ra_vis)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(ra_vis, origin="upper", aspect="auto", cmap="jet")
    ax.axis("off")
    plt.tight_layout(pad=0)


    # 保存到内存缓冲区
    buffer = io.BytesIO()
    plt.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    plt.show()
    # plt.close(fig)

    return buffer


def visradarrastream(ranpypath: str, channel: int = 0, logscale: bool = True,
                     figsize: Tuple[int, int] = (6, 10), format: str = 'png', dpi: int = 300) -> io.BytesIO:
    """
    RA (Range-Azimuth) 可视化 - 基于最新 192x64x20 代码
    支持 .bin 和 .npy，方位角翻转，动态归一化，无坐标无图注
    """
    # 加载数据
    if ranpypath.endswith('.bin'):
        radar_data = np.fromfile(ranpypath, dtype=np.float32).reshape(192, 64, 20)
    else:
        radar_data = np.load(ranpypath, allow_pickle=True)

    # 全通道平均处理（匹配您的最新代码）
    if radar_data.ndim == 3 and radar_data.shape[2] >= 20:
        radar_power = np.abs(radar_data) ** 2  # 功率计算
        radar_data = np.mean(radar_power, axis=2)  # (192,64)
    else:
        radar_data = np.abs(radar_data[..., channel])  # 兼容旧格式

    # 方位角翻转（axis=1）
    radar_data = np.flip(radar_data, axis=1)

    # 动态归一化（3%-97% 分位数，匹配您的 imshow 版本）
    vmin = np.percentile(radar_data, 3)
    vmax = np.percentile(radar_data, 97)
    radar_data = np.clip((radar_data - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # 创建画布
    fig, ax = plt.subplots(figsize=figsize)

    # imshow 可视化（精确匹配您的参数）
    im = ax.imshow(radar_data,
                   origin='lower',
                   aspect='auto',
                   extent=[-31, 32, 0, 191],  # 匹配您的 extent
                   cmap='jet',
                   interpolation='none')

    # 移除所有坐标、标签、标题、颜色条、网格
    ax.axis('off')

    # 保存到 BytesIO
    buffer = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    plt.show()
    plt.close(fig)

    return buffer


def filter_pointcloud(
    points: np.ndarray,
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None,
    z_range: Optional[List[float]] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    根据 XYZ 范围过滤点云.

    Args:
        points: 点云数据 (N, 9)
        x_range: X 坐标范围 [min, max]
        y_range: Y 坐标范围 [min, max]
        z_range: Z 坐标范围 [min, max]
        verbose: 是否打印过滤统计信息

    Returns:
        filtered: 过滤后的点云
    """
    if x_range is None:
        x_range = BEV_X_RANGE
    if y_range is None:
        y_range = BEV_Y_RANGE
    if z_range is None:
        z_range = BEV_Z_RANGE

    if verbose:
        print(f"  BEV 过滤范围:")
        print(f"    X: [{x_range[0]:.1f}, {x_range[1]:.1f}] m")
        print(f"    Y: [{y_range[0]:.1f}, {y_range[1]:.1f}] m")
        print(f"    Z: [{z_range[0]:.1f}, {z_range[1]:.1f}] m")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = (
        (x >= x_range[0]) & (x <= x_range[1]) &
        (y >= y_range[0]) & (y <= y_range[1]) &
        (z >= z_range[0]) & (z <= z_range[1])
    )

    filtered = points[mask]

    if verbose:
        print(f"  过滤统计:")
        print(f"    原始点数: {len(points)}")
        print(f"    保留点数: {len(filtered)}")
        if len(points) > 0:
            print(f"    保留比例: {100*len(filtered)/len(points):.2f}%")

    return filtered

def vis_lidar_os1_stream(
    os1_npy_path: str,
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
) -> io.BytesIO:
    """
    可视化激光雷达 OS1-128 点云.

    K-Radar os1.npy 格式: (N, 9)
    - 字段: x, y, z, intensity, t, reflectivity, ring, ambient, range

    Args:
        os1_npy_path: os1.npy 文件路径
        mode: 可视化模式 ('bev' 俯视图 / '3d' Open3D交互)
        color_by: 颜色映射字段 ('z', 'intensity', 'reflectivity', 'range')
        point_size: 点的大小
        max_points: 最大显示点数（用于降采样）
        bev_filter: 是否应用 BEV 范围过滤
        x_range, y_range, z_range: 自定义 BEV 范围
        figsize: 图像尺寸
        save_path: 保存路径（可选）
        show: 是否显示图像

    Returns:
        buffer,字节流
    """
    pts = np.load(os1_npy_path, allow_pickle=True)

    # 确保是 (N, 9) 格式
    if pts.ndim > 2:
        pts = pts.reshape(-1, pts.shape[-1])

    print(f"  原始点云: {pts.shape[0]} 点")

    # BEV 范围过滤
    if bev_filter:
        pts = filter_pointcloud(pts, x_range, y_range, z_range, verbose=True)
        if len(pts) == 0:
            print("  ⚠️  BEV 过滤后没有剩余点！")
            return pts

    # 提取 xyz
    xyz = pts[:, :3].astype(np.float32)

    # 过滤无效点
    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]
    pts = pts[mask]

    # 降采样
    if max_points and xyz.shape[0] > max_points:
        indices = np.random.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[indices]
        pts = pts[indices]
        print(f"  降采样到: {max_points} 点")

    # 选择颜色字段
    color_map = {
        "z": (xyz[:, 2], "Height (m)"),
        "intensity": (pts[:, 3], "Intensity"),
        "reflectivity": (pts[:, 5], "Reflectivity"),
        "range": (pts[:, 8], "Range (m)")
    }

    if color_by not in color_map:
        color_by = "intensity"



    color_data, color_label = color_map[color_by]

    # 第 220 行附近，在选择 color_data 后添加
    # 强制归一化所有字段
    color_data = (color_data - color_data.min()) / (color_data.max() - color_data.min() + 1e-8)
    color_data = np.clip(color_data, 0, 1)

    # 绘制 BEV
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(-xyz[:, 1], xyz[:, 0], s=point_size, c=color_data,
               cmap="viridis", alpha=0.8)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    plt.tight_layout(pad=0)


    # 保存到内存缓冲区
    buffer = io.BytesIO()
    plt.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    plt.show()
    # plt.close(fig)

    return buffer





# ============ 主程序调用示例 ============
def visualize_to_stream(
    sample_dir: str,
    format: str = 'png',
    dpi: int = 200,
    verbose: bool = False
) -> Dict[str, io.BytesIO]:
    """
    可视化样本并返回图像字节流（用于前端传输）.

    Args:
        sample_dir: 样本目录路径
        format: 图像格式 ('png' 或 'jpg')
        dpi: 图像分辨率
        verbose: 是否打印处理信息

    Returns:
        streams: 包含三个图像字节流的字典
            {
                'camera': BytesIO,
                'radar': BytesIO,
                'lidar': BytesIO
            }

    Examples:
        >>> streams = visualize_to_stream('/path/to/sample')
        >>> # Flask 使用示例
        >>> from flask import send_file
        >>> return send_file(streams['camera'], mimetype='image/png')
    """
    # 检查样本目录
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"样本目录不存在: {sample_dir}")

    mono_path = os.path.join(sample_dir, "mono.jpg")
    ra_path = os.path.join(sample_dir, "ra.npy")
    os1_path = os.path.join(sample_dir, "os1.npy")

    # 检查文件
    missing_files = []
    for name, path in [("mono.jpg", mono_path), ("ra.npy", ra_path), ("os1.npy", os1_path)]:
        if not os.path.exists(path):
            missing_files.append(name)

    if missing_files:
        raise FileNotFoundError(
            f"样本目录缺少文件: {', '.join(missing_files)}\n目录: {sample_dir}"
        )

    if verbose:
        print(f"可视化样本: {os.path.basename(sample_dir)}")

    # 生成三个图像的字节流
    streams = {}

    if verbose:
        print("[1/3] 生成相机图像...")
    streams['camera'] = vis_image_stream(mono_path, format=format, dpi=dpi)

    if verbose:
        print("[2/3] 生成雷达热力图...")
    streams['radar'] = vis_radar_ra_stream(ra_path, channel=0, format=format, dpi=dpi)
    # streams['radar'] = visradarrastream(ra_path, format='png', dpi=300)

    if verbose:
        print("[3/3] 生成激光雷达点云...")
    streams['lidar'] = vis_lidar_os1_stream(
        os1_path,
        color_by="reflectivity",
        bev_filter=True,
        format=format,
        dpi=dpi,
        verbose=verbose
    )

    if verbose:
        print("✓ 完成")

    return streams



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('DPRT 单样本推理（支持全数据集）')
#     parser.add_argument('--src', type=str, default="/mnt/disk1/xiaxue/datasets/kradar_processed2",
#                         help="数据集根目录 e.g. /mnt/disk1/xiaxue/datasets/kradar_processed2")
#     parser.add_argument('--cfg', type=str,default="/mnt/disk1/xiaxue/code/new/dpft_v4/config/kradar.json",
#                         help="配置文件 e.g. /mnt/.../kradar.json")
#     parser.add_argument('--checkpoint', type=str,default="/mnt/disk1/xiaxue/code/DPFT-main/DPFT-main/log/20250929-121010-276/checkpoints/20250929-121010-276_checkpoint_0103.pt",
#                         help="模型checkpoint路径")
#     parser.add_argument('--sample_dir', type=str, default="/mnt/disk1/xiaxue/datasets/kradar_processed2/test/12/00725_00689",
#                         help="任意样本目录 e.g. /mnt/.../kradar_processed2/train/1/00033_00001")
#     parser.add_argument('--split', type=str, default=None,
#                         choices=['train', 'val', 'test'],
#                         help="指定split（不填自动推断）")
#     parser.add_argument('--dst', type=str, default="/mnt/disk1/xiaxue/code/new/dpft_v4/log/test_res",
#                         help="输出目录")
#
#     args = parser.parse_args()
#     main(args.src, args.cfg, args.checkpoint, args.dst, args.sample_dir, args.split)
# ============ 使用示例 ============
# ============ 使用示例 ============
if __name__ == "__main__":
    sample_dir = "/mnt/disk3/xiaxue/dataset/k_radar_v1/test/57/00218_00213"

    # 示例 1: 返回 BytesIO（适合 Flask send_file）
    print("\n=== 示例 1: BytesIO 流 ===")
    streams = visualize_to_stream(sample_dir, verbose=True)
    print(f"类型: {type(streams['camera'])}")
    print(f"大小: camera={len(streams['camera'].getvalue())} bytes")
    print(f"大小: radar ={len(streams['radar'].getvalue())} bytes")
    print(f"大小: lidar={len(streams['lidar'].getvalue())} bytes")

    # 示例 2: 单独可视化某个传感器
    # vis_radar_ra(os.path.join(sample_dir, "ra.npy"), channel=0)
    # vis_lidar_os1(os.path.join(sample_dir, "os1.npy"), mode="3d", color_by="intensity")