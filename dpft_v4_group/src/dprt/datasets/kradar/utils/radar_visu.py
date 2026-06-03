import numpy as np
import matplotlib.pyplot as plt


def visualize_radar_data_by_pcolormesh(file_path, save_path='radar_ra_00034.png'):
    """
    毫米波雷达多通道平均可视化（含方位角翻转）

    参数：
    file_path: .npy文件路径
    save_path: 图像保存路径
    """
    # 加载数据
    radar_data = np.load(file_path)  # 形状(192,64,20)

    # 全通道平均处理
    radar_data = np.mean(radar_data, axis=2)  # (192,64)

    # 方位角处理
    radar_data = np.flip(radar_data, axis=1)  # 沿方位维度翻转
    azimuth_angles = np.linspace(-31, 32, 64)  # 方位角范围
    range_indices = np.arange(192)  # 距离索引

    # 创建坐标网格
    X, Y = np.meshgrid(azimuth_angles, range_indices)

    # 动态归一化（排除极端值）
    # vmin = np.percentile(radar_data, 1)
    # vmax = np.percentile(radar_data, 99)
    # radar_data = np.clip((radar_data - vmin) / (vmax - vmin), 0, 1)

    # 创建画布
    plt.figure(figsize=(6, 10))

    # 绘制热力图
    ax = plt.gca()
    im = ax.pcolormesh(X, Y, radar_data,  # 使用坐标网格
                       cmap='jet',
                       shading='auto',
                       vmin=radar_data.min(), vmax=radar_data.max())
    # min:0     max:26.38520744031769

    # 设置坐标轴标签
    ax.set_xlabel('Azimuth Index ', fontsize=12)
    ax.set_ylabel('Range Index', fontsize=12)
    ax.set_title('Radar RA', fontsize=14)

    # 设置坐标轴刻度
    ax.set_xticks(np.linspace(-30, 30, 7))  # 每10度一个主刻度
    ax.set_yticks(np.linspace(0, 190, 6))  # 每38个距离单元一个刻度

    # 添加颜色条
    cbar = plt.colorbar(im, pad=0.02, aspect=40)
    cbar.set_label('Signal Intensity', fontsize=12)

    # 优化布局并保存
    plt.grid(linestyle='--', alpha=0.3, color='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def visualize_radar_data_by_imshow(file_path, save_path='radar_ra_00034_imshow.png'):
    """
    毫米波雷达多通道平均可视化（优化imshow版本）

    参数：
    file_path: .npy文件路径
    save_path: 图像保存路径
    """
    # 加载数据（保持原始结构）
    radar_data = np.load(file_path)  # (192,64,20)
    radar_data = np.mean(radar_data, axis=2)  # (192,64)

    # 方位角翻转处理（保持与pcolormesh一致）
    radar_data = np.flip(radar_data, axis=1)

    # 动态归一化（匹配原逻辑）
    vmin = np.percentile(radar_data, 3)
    vmax = np.percentile(radar_data, 97)
    normalized = np.clip((radar_data - vmin) / (vmax - vmin), 0, 1)

    # 创建画布（保持原尺寸）
    plt.figure(figsize=(6, 10))

    # 关键改进：imshow参数设置
    im = plt.imshow(normalized,
                    origin='lower',  # 强制原点在左下
                    aspect='auto',  # 自动宽高比
                    extent=[-31, 32, 0, 191],  # X/Y轴范围精确映射
                    cmap='jet',
                    interpolation='none')  # 禁用插值保持像素清晰

    # 坐标轴标签（保持原样式）
    plt.xlabel('Azimuth Index ', fontsize=12)
    plt.ylabel('Range Index', fontsize=12)
    plt.title('Radar RA', fontsize=14)

    # 刻度设置改进（精确对齐数据边界）
    plt.xticks(np.linspace(-30, 30, 7),
               labels=[-30, -20, -10, 0, 10, 20, 30])  # 显示整数刻度
    plt.yticks(np.linspace(0, 190, 6))

    # 颜色条同步（保持原参数）
    cbar = plt.colorbar(im, pad=0.02, aspect=40)
    cbar.set_label('Signal Intensity', fontsize=12)

    # 网格线增强（保持视觉效果一致性）
    plt.grid(True, which='both',
             linestyle='--', alpha=0.3,
             color='white', linewidth=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    data_path = '/mnt/6feca051-3223-4732-996b-597606847bb2/xiaxue/datasets/kradar_processed2/test/42/00462_00453/ra.npy'
    visualize_radar_data_by_pcolormesh(file_path=data_path, save_path='ra_000034_pcolormesh.png')
    visualize_radar_data_by_imshow(file_path=data_path, save_path='ra_000034_imshow.png')