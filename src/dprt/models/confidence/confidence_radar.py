import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dprt.training.assigner import SimpleHungarianAssigner
import math
from dprt.models.confidence.feature_selector import MultiScaleFeatureSelector

from torchvision.models import resnet50
from collections import OrderedDict
class ModalityLevelDynamics(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # 创建模型（不自动加载权重）
        backbone = resnet50(weights=None)
        
         # 添加调整层来处理6通道输入
        self.adjustment_layer = nn.Conv2d(6, out_channels=3, kernel_size=(1, 1),
                                         stride=1, padding=0, bias=False)
        self.conv1 = backbone.conv1  # 保持标准的3输入通道
        
        # 注意：不再自动加载预训练权重，权重将在训练时从主模型复制
        # 移除了原来加载预训练权重的代码
        
        # 提取其他层（保持不变）
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels  
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels
        
        
        # 多尺度特征融合
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(256, 128, kernel_size=1),   # layer1
            nn.Conv2d(512, 128, kernel_size=1),   # layer2
            nn.Conv2d(1024, 128, kernel_size=1),  # layer3
            nn.Conv2d(2048, 128, kernel_size=1),  # layer4
        ])
        
        self.fusion_weights = nn.Parameter(torch.ones(4))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 初始化新增层
        self._initialize_weights()

    
    def _initialize_weights(self):
        """初始化新增层的权重"""
        for conv in self.scale_convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)


    def forward(self, x, model_selectors_radar_bev = None):
        """
        前向传播 - 使用多尺度特征
        
        参数：
            x: 6通道毫米波图像 tensor, shape=(batch_size, H, W, 6)
            model_selectors_camera: 稀疏特征选择器
        """
        # 调整维度: [B, H, W, 6] -> [B, 6, H, W]
        x = x.permute(0, 3, 1, 2)

          # 先通过调整层将6通道转换为3通道
        x = self.adjustment_layer(x)
        
        # 使用标准的3通道conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        feat1 = self.layer1(x)  # [B, 256, H1, W1]
        feat2 = self.layer2(feat1)  # [B, 512, H2, W2]
        feat3 = self.layer3(feat2)  # [B, 1024, H3, W3]  
        feat4 = self.layer4(feat3)  # [B, 2048, H4, W4]
        
         # 应用稀疏特征选择器（如果提供）
        if model_selectors_radar_bev is not None:
            # 将特征转换为稀疏选择器所需的格式 OrderedDict {'1': feat1, '2': feat2, ...}
            multi_scale_features = OrderedDict([
                ('1', feat1.permute(0, 2, 3, 1)),  # [B, H1, W1, 256]
                ('2', feat2.permute(0, 2, 3, 1)),  # [B, H2, W2, 512]
                ('3', feat3.permute(0, 2, 3, 1)),  # [B, H3, W3, 1024]
                ('4', feat4.permute(0, 2, 3, 1))   # [B, H4, W4, 2048]
            ])
            
            # 应用稀疏选择器
            selector_outputs = model_selectors_radar_bev(multi_scale_features, training=False)
            gated_features = selector_outputs['gated_features']
            
            # 将特征转回原来的格式 [B, C, H, W]
            feats = [gated_features[key].permute(0, 3, 1, 2) for key in ['1', '2', '3', '4']]
        else:
            # 如果没有提供稀疏选择器，使用原始特征
            feats = [feat1, feat2, feat3, feat4]
        
        # 对齐和融合多尺度特征
        aligned = []
        
        # 统一缩放到最小尺寸
        target_size = feats[-1].shape[2:]  # 使用layer4的尺寸
        
        for i, (feat, conv) in enumerate(zip(feats, self.scale_convs)):
            x = conv(feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(x)
        
        # 加权融合
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused_feat = sum(w * feat for w, feat in zip(weights, aligned))
        
        # 全局平均池化 + 分类
        x = fused_feat.mean(dim=[2, 3])  # [B, 128]
        confidence = self.classifier(x)  # [B, 1]
        
        return confidence
    
    def compute_matched_tcp(self, class_probs, targets, indices):
        """
        使用IoU计算TCP值 - 改进版本
        
        参数:
            class_probs: 预测结果字典列表，包含center、size、angle、class等
            targets: 真实标签字典列表
            indices: 匹配索引元组
            
        返回:
            matched_tcp: 基于IoU的TCP值，shape=(batch_size, 1)
        """
        # 处理indices格式
        if isinstance(indices, list) and len(indices) == 2 and isinstance(indices[0], list):
            pred_indices_list, target_indices_list = indices
        else:
            pred_indices_list, target_indices_list = indices
        
        matched_tcp_values = []
        device = class_probs[0]["class"].device if isinstance(class_probs, list) else class_probs.device
        batch_size = min(len(class_probs), len(targets))
        
        for batch_idx in range(batch_size):
            pred_indices = pred_indices_list[batch_idx]
            target_indices = target_indices_list[batch_idx]
            
            if len(pred_indices) > 0 and len(target_indices) > 0:
                # 获取匹配的预测和真实目标
                pred_centers = class_probs[batch_idx]["center"][pred_indices]  # [N, 3]
                pred_sizes = class_probs[batch_idx]["size"][pred_indices]      # [N, 3]
                pred_angles = class_probs[batch_idx]["angle"][pred_indices]    # [N, 2]
                pred_cls_probs = class_probs[batch_idx]["class"][pred_indices] # [N, num_classes]
                
                gt_centers = targets[batch_idx]["gt_center"][target_indices]   # [N, 3]
                gt_sizes = targets[batch_idx]["gt_size"][target_indices]       # [N, 3]
                gt_angles = targets[batch_idx]["gt_angle"][target_indices]     # [N, 2]
                gt_classes = targets[batch_idx]["gt_class"][target_indices]    # [N, num_classes]
                
                # 1. 计算3D IoU
                iou_3d = self.compute_3d_iou_pairs(
                    pred_centers, pred_sizes, pred_angles,
                    gt_centers, gt_sizes, gt_angles
                )  # [N]
                
                # 2. 计算分类质量 (分类概率与真实标签的匹配度)
                cls_quality = torch.sum(gt_classes * pred_cls_probs, dim=1)  # [N]
                
                # 3. 综合TCP计算：IoU为主导，分类为辅助
                # 基于IoU-Net和3DIoUMatch的思想
                comprehensive_tcp = (
                    0.9 * iou_3d +           # IoU是主要的定位质量指标
                    0.1 * cls_quality        # 分类置信度作为辅助
                )
                
                # 对所有匹配对取平均
                avg_tcp = torch.mean(comprehensive_tcp, dim=0, keepdim=True).unsqueeze(0)  # [1, 1]
                
            else:
                # 没有匹配目标时返回0
                avg_tcp = torch.tensor([[0.0]], device=device)
            
            matched_tcp_values.append(avg_tcp)
        
        return torch.cat(matched_tcp_values, dim=0)  # [batch_size, 1]

    def compute_3d_iou_pairs(self, pred_centers, pred_sizes, pred_angles, 
                        gt_centers, gt_sizes, gt_angles):
        """
        计算配对的3D IoU (每个预测框与对应的真实框)
        
        参数:
            pred_centers: [N, 3] 预测中心点
            pred_sizes: [N, 3] 预测尺寸
            pred_angles: [N, 2] 预测角度 (sin, cos)
            gt_centers: [N, 3] 真实中心点  
            gt_sizes: [N, 3] 真实尺寸
            gt_angles: [N, 2] 真实角度 (sin, cos)
            
        返回:
            iou_values: [N] 每对框的IoU值
        """
        # 将sin,cos转换为角度
        pred_yaw = torch.atan2(pred_angles[:, 0], pred_angles[:, 1])  # [N]
        gt_yaw = torch.atan2(gt_angles[:, 0], gt_angles[:, 1])        # [N]
        
        # 为3D IoU计算准备数据
        # 添加batch维度以兼容现有的3D IoU函数
        pred_centers_batch = pred_centers.unsqueeze(0)  # [1, N, 3]
        pred_sizes_batch = pred_sizes.unsqueeze(0)      # [1, N, 3]  
        pred_yaw_batch = pred_yaw.unsqueeze(0)          # [1, N]
        
        gt_centers_batch = gt_centers.unsqueeze(0)      # [1, N, 3]
        gt_sizes_batch = gt_sizes.unsqueeze(0)          # [1, N, 3]
        gt_yaw_batch = gt_yaw.unsqueeze(0)              # [1, N]
        
        # 计算3D bounding box的8个角点
        pred_corners = self.get_box_corners(pred_centers_batch, pred_sizes_batch, pred_yaw_batch)  # [1, N, 8, 3]
        gt_corners = self.get_box_corners(gt_centers_batch, gt_sizes_batch, gt_yaw_batch)          # [1, N, 8, 3]
        
        # 计算IoU矩阵然后取对角线元素 (配对IoU)
        iou_matrix = self.giou3d(pred_corners, gt_corners)  # [1, N, N]
        iou_values = torch.diag(iou_matrix[0])              # [N] - 取对角线元素
        
        return iou_values
    
    def get_box_corners(self, centers, sizes, yaws):
        """
        计算3D边界框的8个角点坐标
        
        参数:
            centers: [B, N, 3] 中心点坐标 (x, y, z)
            sizes: [B, N, 3] 尺寸 (length, width, height)
            yaws: [B, N] 偏航角（绕z轴旋转）
            
        返回:
            corners: [B, N, 8, 3] 8个角点的坐标
        """
        B, N, _ = centers.shape
        
        # 定义单位立方体的8个角点（在标准坐标系中）
        # 假设长度沿x轴，宽度沿y轴，高度沿z轴
        corners_norm = torch.tensor([
            [0.5, 0.5, 0.5],   # 0: 右前上
            [-0.5, 0.5, 0.5],  # 1: 左前上
            [-0.5, -0.5, 0.5], # 2: 左后上
            [0.5, -0.5, 0.5],  # 3: 右后上
            [0.5, 0.5, -0.5],  # 4: 右前下
            [-0.5, 0.5, -0.5], # 5: 左前下
            [-0.5, -0.5, -0.5],# 6: 左后下
            [0.5, -0.5, -0.5]  # 7: 右后下
        ], device=centers.device, dtype=centers.dtype)  # [8, 3]
        
        # 扩展维度以进行广播
        corners_norm = corners_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, 8, 3]
        
        # 将单位立方体角点缩放到实际尺寸
        sizes_expanded = sizes.unsqueeze(2)  # [B, N, 1, 3]
        corners = corners_norm * sizes_expanded  # [B, N, 8, 3]
        
        # 应用偏航角旋转（绕z轴）
        cos_yaw = torch.cos(yaws).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        sin_yaw = torch.sin(yaws).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        
        # 2D旋转矩阵应用到x,y坐标
        corners_x = corners[..., 0:1] * cos_yaw - corners[..., 1:2] * sin_yaw  # [B, N, 8, 1]
        corners_y = corners[..., 0:1] * sin_yaw + corners[..., 1:2] * cos_yaw  # [B, N, 8, 1]
        corners_z = corners[..., 2:3]  # [B, N, 8, 1]
        
        # 组合旋转后的坐标
        corners_rotated = torch.cat([corners_x, corners_y, corners_z], dim=-1)  # [B, N, 8, 3]
        
        # 平移到实际中心位置
        centers_expanded = centers.unsqueeze(2)  # [B, N, 1, 3]
        corners_final = corners_rotated + centers_expanded  # [B, N, 8, 3]
        
        return corners_final
    
    def giou3d(self, pred_corners, gt_corners):
        """
        计算完整的3D IoU矩阵
        
        参数:
            pred_corners: [B, M, 8, 3] 预测框角点
            gt_corners: [B, N, 8, 3] 真实框角点
            
        返回:
            iou_matrix: [B, M, N] IoU矩阵
        """
        B, M, _, _ = pred_corners.shape
        _, N, _, _ = gt_corners.shape
        
        # 初始化IoU矩阵
        iou_matrix = torch.zeros(B, M, N, device=pred_corners.device, dtype=pred_corners.dtype)
        
        # 计算每对框的IoU
        for i in range(M):
            for j in range(N):
                for b in range(B):
                    # 获取单个框的角点
                    pred_box_corners = pred_corners[b, i]  # [8, 3]
                    gt_box_corners = gt_corners[b, j]      # [8, 3]
                    
                    # 计算边界框
                    pred_min = torch.min(pred_box_corners, dim=0)[0]  # [3]
                    pred_max = torch.max(pred_box_corners, dim=0)[0]  # [3]
                    gt_min = torch.min(gt_box_corners, dim=0)[0]      # [3]
                    gt_max = torch.max(gt_box_corners, dim=0)[0]      # [3]
                    
                    # 计算交集体积
                    inter_min = torch.max(pred_min, gt_min)  # [3]
                    inter_max = torch.min(pred_max, gt_max)  # [3]
                    
                    inter_vol = 0.0
                    if torch.all(inter_max > inter_min):
                        inter_vol = torch.prod(inter_max - inter_min)
                    
                    # 计算并集体积
                    pred_vol = torch.prod(pred_max - pred_min)
                    gt_vol = torch.prod(gt_max - gt_min)
                    union_vol = pred_vol + gt_vol - inter_vol
                    
                    # 计算IoU
                    iou_matrix[b, i, j] = inter_vol / (union_vol + 1e-8)
        
        return iou_matrix
