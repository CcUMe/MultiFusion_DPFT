import pandas as pd
from matplotlib import pyplot as plt

from dprt.utils.calib import show_projected_point_cloud
from dprt.utils.calib import get_matrices_from_dict_calib
import cv2
import numpy as np
import open3d as o3d


def plot_3d_bbox(img, label_df, list_params):
    img_size, intrinsics, distortion, T_ldr2cam = list_params
    T_cam2pix = np.insert(np.insert(intrinsics, 3, [0, 0, 0], axis=1), 3, [0, 0, 0, 1], axis=0)
    T_ldr2pix = T_cam2pix @ np.insert(T_ldr2cam, 3, [0, 0, 0, 1], axis=0)

    for _, obj in label_df.iterrows():
        # 生成边界框顶点
        l, w, h = 2 * obj['half_length'], 2 * obj['half_width'], 2 * obj['half_height']
        yaw = np.deg2rad(obj['yaw'])

        # 局部坐标系下的8个顶点
        corners = np.array([[l / 2, w / 2, h / 2], [l / 2, -w / 2, h / 2],
                            [-l / 2, -w / 2, h / 2], [-l / 2, w / 2, h / 2],
                            [l / 2, w / 2, -h / 2], [l / 2, -w / 2, -h / 2],
                            [-l / 2, -w / 2, -h / 2], [-l / 2, w / 2, -h / 2]])

        # 偏航角
        # rot_matrix = np.array([[np.cos(yaw), 0, np.sin(yaw)],
        #                        [0, 1, 0],
        #                        [-np.sin(yaw), 0, np.cos(yaw)]])
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])
        rotated = (rot_matrix @ corners.T).T

        # 转换为全局坐标系
        global_pts = rotated + [obj['x'], obj['y'], obj['z']]

        # 坐标变换
        homogenous_pts = np.insert(global_pts, 3, 1, axis=1).T
        projected = T_ldr2pix @ homogenous_pts
        projected[:2] /= projected[2]  # 透视除法

        # 过滤可见点
        valid = (projected[0] >= 0) & (projected[0] < img_size[0]) & \
                (projected[1] >= 0) & (projected[1] < img_size[1]) & \
                (projected[2] > 0)

        if np.any(valid):
            # 绘制边界框连线
            edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                     (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]

            for start, end in edges:
                x1, y1 = projected[0, start], projected[1, start]
                x2, y2 = projected[0, end], projected[1, end]
                if valid[start] and valid[end]:
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return img


# 修改主函数部分
if __name__ == '__main__':
    # img_path='/mnt/mobile_disk/kradar/20/cam-front/cam-front_00001.png'
    # label_path = '/mnt/mobile_disk/kradar/20/info_label/00034_00001.txt'

    # img_path='/mnt/mobile_disk/kradar/27/cam-front/cam-front_01569.png'
    # label_path = '/mnt/mobile_disk/kradar/27/info_label/00528_00524.txt'

    # img_path = '/mnt/mobile_disk/kradar/20/cam-front/cam-front_00244.png'
    # label_path = '/mnt/mobile_disk/kradar/20/info_label/00115_00082.txt'

    # img_path = '/mnt/mobile_disk/kradar/13/cam-front/cam-front_00186.png'
    # label_path = '/mnt/mobile_disk/kradar/13/info_label/00101_00062.txt'

    # img_path = '/mnt/mobile_disk/kradar/20/cam-front/cam-front_00250.png'
    # label_path = '/mnt/mobile_disk/kradar/20/info_label/00117_00084.txt'

    # img_path = '/mnt/mobile_disk/kradar/51/cam-front/cam-front_00689.png'
    # label_path = '/mnt/mobile_disk/kradar/51/info_label/00240_00234.txt'

    # 00014_00012_00035_00012_00028
    # img_path = '/mnt/mobile_disk/kradar/22/cam-front/cam-front_00035.png'
    # label_path = '/mnt/mobile_disk/kradar/22/info_label_v1_1/00014_00012.txt'

    img_path = '/mnt/mobile_disk/kradar/52/cam-front/cam-front_01635.png'
    label_path = '/mnt/mobile_disk/kradar/52/info_label_v2/00555_00553.txt'
    # label_path = '/mnt/mobile_disk/kradar/11/info_label_v2/00743_00710.txt'

    label_df = pd.read_csv(label_path, skiprows=1, header=None,
                           names=['*', 'index','category', 'x', 'y', 'z', 'yaw',
                                  'half_length', 'half_width', 'half_height'])

    # 加载图片
    img = cv2.imread(img_path)
    img = img[:,:1280]

    list_params = get_matrices_from_dict_calib()
    # 投影3D边界框
    bbox_img = plot_3d_bbox(img.copy(), label_df, list_params)
    # cv2.imwrite('20_00101_00062_00186.png', bbox_img)
    # cv2.imwrite('27_00528_00524_01569.png', bbox_img)
    # cv2.imwrite('20_00115_00082_00244.png', bbox_img)
    # cv2.imwrite('13_00101_00062_00186.png', bbox_img)
    # cv2.imwrite('20_00117_00084_00250.png', bbox_img)
    cv2.imwrite('52_00014_00012_00035.png', bbox_img)
    plt.figure(figsize=(12.8, 7.2), dpi=100)
    plt.imshow(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()



