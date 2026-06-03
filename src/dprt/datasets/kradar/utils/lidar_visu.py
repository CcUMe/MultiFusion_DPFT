# filtered_lidar_visu.py - 裁剪并可视化点云（使用反射强度着色）
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# BEV范围（来自kradar.json配置）
BEV_X_RANGE = [0.0, 20.0]
BEV_Y_RANGE = [-6.4, 6.4]
BEV_Z_RANGE = [-2.0, 6.0]

def filter_pointcloud(points, x_range=None, y_range=None, z_range=None):
    """根据XYZ范围过滤点云"""

    if x_range is None:
        x_range = BEV_X_RANGE
    if y_range is None:
        y_range = BEV_Y_RANGE
    if z_range is None:
        z_range = BEV_Z_RANGE

    print(f"\n过滤范围:")
    print(f"  X: {x_range}")
    print(f"  Y: {y_range}")
    print(f"  Z: {z_range}")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    mask = (
        (x >= x_range[0]) & (x <= x_range[1]) &
        (y >= y_range[0]) & (y <= y_range[1]) &
        (z >= z_range[0]) & (z <= z_range[1])
    )

    filtered = points[mask]

    print(f"\n过滤统计:")
    print(f"  原始点数: {len(points)}")
    print(f"  保留点数: {len(filtered)}")
    print(f"  保留比例: {100*len(filtered)/len(points):.2f}%")

    return filtered

def visualize_topview(lidar_path, filter_points=True, x_range=None, y_range=None, z_range=None):
    """可视化点云俯视图（使用反射强度着色）"""

    # 加载点云数据
    print(f"\n正在加载: {lidar_path}")

    if not os.path.exists(lidar_path):
        print(f"❌ 文件不存在: {lidar_path}")
        return

    try:
        points = np.load(lidar_path)
        print(f"✅ 成功加载点云")
        print(f"   形状: {points.shape}")
        print(f"   点数: {points.shape[0]}")

        if len(points.shape) != 2 or points.shape[1] < 3:
            print(f"❌ 点云格式错误，期望 (N, 3+) 得到 {points.shape}")
            return

        # 检查是否有反射强度列
        has_intensity = points.shape[1] >= 4
        print(f"   包含反射强度: {'是' if has_intensity else '否'}")

        # 过滤点云
        if filter_points:
            points = filter_pointcloud(points, x_range, y_range, z_range)
            if len(points) == 0:
                print("❌ 过滤后没有剩余点")
                return

        # 提取XYZ坐标
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        print(f"\n(过滤后)点云统计:")
        print(f"  X范围: [{x.min():.2f}, {x.max():.2f}] m")
        print(f"  Y范围: [{y.min():.2f}, {y.max():.2f}] m")
        print(f"  Z范围: [{z.min():.2f}, {z.max():.2f}] m")

        # 提取反射强度（如果有）
        if has_intensity:
            intensity = points[:, 3]
            print(f"  反射强度范围: [{intensity.min():.2f}, {intensity.max():.2f}]")
            color_data = intensity
            color_label = '反射强度 (Intensity)'
        else:
            print("  ⚠️ 无反射强度数据，将使用Z值着色")
            color_data = z
            color_label = '高度 Z (m)'

        distances = np.sqrt(x**2 + y**2)
        print(f"  距离范围: [{distances.min():.2f}, {distances.max():.2f}] m")

        # 创建俯视图
        fig, ax = plt.subplots(figsize=(14, 6))

        # 使用反射强度或Z值着色
        scatter = ax.scatter(x, y, c=color_data, s=1.0, cmap='viridis', alpha=0.8)

        # 添加边界框
        if filter_points:
            rect_x = x_range if x_range else BEV_X_RANGE
            rect_y = y_range if y_range else BEV_Y_RANGE

            ax.plot([rect_x[0], rect_x[1], rect_x[1], rect_x[0], rect_x[0]],
                   [rect_y[0], rect_y[0], rect_y[1], rect_y[1], rect_y[0]],
                   'r--', linewidth=2, label=f'FOV边界')
            ax.legend(fontsize=12)

        ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
        ax.set_title('LiDAR 点云俯视图 (BEV)', fontsize=16, fontweight='bold')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        # 颜色条（标签根据实际着色数据调整）
        cbar = plt.colorbar(scatter, ax=ax, label=color_label)
        cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()

        # 保存图像
        filename = os.path.basename(lidar_path)
        color_type = 'intensity' if has_intensity else 'height'
        output_name = f"topview_xx{color_type}_{filename.replace('.npy', '')}.png"
        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        print(f"\n✅ 可视化已保存: {output_name}")

        plt.show()

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()

def visualize_directory(dir_path, sensor='os1', filter_points=True,
                       x_range=None, y_range=None, z_range=None):
    """可视化目录下的特定传感器数据"""

    lidar_file = os.path.join(dir_path, f"{sensor}.npy")

    if not os.path.exists(lidar_file):
        print(f"❌ 未找到文件: {lidar_file}")
        print(f"   目录下的文件:")
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                print(f"     - {f}")
        return

    visualize_topview(lidar_file, filter_points, x_range, y_range, z_range)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='点云俯视图可视化 - 支持BEV范围过滤（使用反射强度着色）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
使用示例:

1. 使用默认BEV范围 ({BEV_X_RANGE}, {BEV_Y_RANGE}, {BEV_Z_RANGE}):
   python filtered_lidar_visu.py --file /path/to/os1.npy
   python filtered_lidar_visu.py --dir /path/to/sample/dir

2. 不过滤，显示所有点:
   python filtered_lidar_visu.py --file /path/to/os1.npy --no-filter

3. 自定义范围:
   python filtered_lidar_visu.py --file /path/to/os1.npy --x_range 0 100 --y_range -10 10

4. 根据你的需求（kradar BEV范围）:
   python filtered_lidar_visu.py --dir /mnt/.../test/42/00462_00453

注意: 
- 如果点云包含第4列，将自动使用反射强度着色
- 如果点云只有XYZ（3列），将使用高度Z值着色并显示警告
        """
    )

    parser.add_argument('--file', type=str, help='点云文件路径 (.npy)')
    parser.add_argument('--dir', type=str, help='样本目录路径')
    parser.add_argument('--sensor', type=str, default='os1', choices=['os1', 'os2'],
                       help='传感器类型 (默认: os1)')

    parser.add_argument('--no-filter', action='store_true',
                       help='不过滤点云，显示所有点')

    parser.add_argument('--x_range', nargs=2, type=float, default=None,
                       help=f'X范围 (默认: {BEV_X_RANGE})')
    parser.add_argument('--y_range', nargs=2, type=float, default=None,
                       help=f'Y范围 (默认: {BEV_Y_RANGE})')
    parser.add_argument('--z_range', nargs=2, type=float, default=None,
                       help=f'Z范围 (默认: {BEV_Z_RANGE})')

    args = parser.parse_args()

    filter_points = not args.no_filter

    if args.file:
        visualize_topview(args.file, filter_points,
                         args.x_range, args.y_range, args.z_range)
    elif args.dir:
        visualize_directory(args.dir, args.sensor, filter_points,
                           args.x_range, args.y_range, args.z_range)
    else:
        print("❌ 请指定 --file 或 --dir 参数")
        parser.print_help()
