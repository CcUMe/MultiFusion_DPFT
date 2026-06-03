"""统一的BEV投影配置文件，基于kradar.json中的FOV数据"""

# 基于kradar.json中FOV的保守BEV配置
BEV_X_RANGE = [0.0, 20.0]      # 与kradar.json中x完全一致
BEV_Y_RANGE = [-6.4, 6.4]     # 与kradar.json中y完全一致
BEV_Z_RANGE = [-2.0, 6.0]     # 与kradar.json中z完全一致

# BEV网格分辨率
BEV_RESOLUTION = 0.2  # 20cm分辨率

# BEV图像尺寸计算
BEV_WIDTH = int((BEV_X_RANGE[1] - BEV_X_RANGE[0]) / BEV_RESOLUTION)   # 360像素
BEV_HEIGHT = int((BEV_Y_RANGE[1] - BEV_Y_RANGE[0]) / BEV_RESOLUTION)  # 64像素

# 中心点偏移（用于投影矩阵计算）
BEV_X_CENTER = (BEV_X_RANGE[1] + BEV_X_RANGE[0]) / 2  # 36.0
BEV_Y_CENTER = (BEV_Y_RANGE[1] + BEV_Y_RANGE[0]) / 2  # 0.0

# print(f"BEV配置加载: {BEV_WIDTH}x{BEV_HEIGHT}像素, 覆盖范围X:{BEV_X_RANGE}, Y:{BEV_Y_RANGE}")
