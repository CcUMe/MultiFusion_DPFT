"""
LiDAR BEV 投影配置，基于 K-Radar LiDAR 标注范围
"""
# LiDAR BEV 范围：0–120m 前向，±40m 侧向
BEV_X_RANGE = [0.0, 72.0]
BEV_Y_RANGE = [-6.4, 6.4]
BEV_Z_RANGE = [-2.0, 6.0]
# 分辨率 25cm → 480×320
BEV_RESOLUTION = 0.20
BEV_WIDTH  = int((BEV_X_RANGE[1] - BEV_X_RANGE[0]) / BEV_RESOLUTION)
BEV_HEIGHT = int((BEV_Y_RANGE[1] - BEV_Y_RANGE[0]) / BEV_RESOLUTION)
# 可选中心偏移
BEV_X_CENTER = (BEV_X_RANGE[1] + BEV_X_RANGE[0]) / 2  # 60.0
BEV_Y_CENTER = (BEV_Y_RANGE[1] + BEV_Y_RANGE[0]) / 2  # 0.0


# LiDAR点云归一化参数
max_intensity = 255.0
min_intensity = 0.0
max_height = BEV_Z_RANGE[1]  # 6.0
min_height = BEV_Z_RANGE[0]  # -2.0

# 点云密度参数
max_point_density = 100
min_point_density = 0