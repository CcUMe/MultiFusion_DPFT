import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class DifferentiableSparseGate(nn.Module):
    """
    可微分稀疏门控模块
    实现基于Gumbel-Softmax的可微分二值化门控
    """
    def __init__(self, temperature=0.1, gate_bias_init=-1.0):
        super().__init__()
        self.temperature = temperature
        # 初始化偏置为负值，鼓励初始稀疏性
        self.gate_bias = nn.Parameter(torch.tensor([gate_bias_init]))
        
    def forward(self, importance_weights, training):
        """
        参数:
            importance_weights: 重要性权重 [batch_size, feature_dim]
            training: 训练模式标志
        """
        # 1. 应用可学习偏置调整稀疏度
        logits = torch.log(importance_weights + 1e-8) - torch.log(1 - importance_weights + 1e-8)
        adjusted_logits = logits + self.gate_bias
        
        if training:
            # 2. 训练时使用Gumbel-Softmax松弛
            # 生成Gumbel噪声
            uniform = torch.rand_like(adjusted_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
            
            # 添加噪声并应用温度参数
            noisy_logits = (adjusted_logits + gumbel_noise) / self.temperature
            
            # 3. 通过sigmoid获得松弛的门控值
            relaxed_gate = torch.sigmoid(noisy_logits)
            
            # 4. 直通估计器：前向用硬门控，反向用松弛门控的梯度
            with torch.no_grad():
                hard_gate = (relaxed_gate > 0.5).float()
            
            # 直通技巧
            gate = hard_gate - relaxed_gate.detach() + relaxed_gate
            
        else:
            # 5. 推理时使用硬门控
            gate = (adjusted_logits > 0).float()
        
        return gate

class MultiScaleFeatureSelector(nn.Module):
    """
    多尺度动态特征选择模块
    专门处理来自不同尺度（如FPN特征）的特征图，保持[B,H,W,C]形状不变
    """
    def __init__(self, feature_channels, hidden_ratio=2, temperature=0.1, 
                 sparsity_lambda=0.01, dropout=0.1):
        """
        参数:
            feature_channels: 各尺度特征的通道数列表 [channel1, channel2, ...]
            hidden_ratio: 隐藏层维度倍数
            temperature: Gumbel-Softmax温度参数
            sparsity_lambda: 稀疏性损失系数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.feature_channels = feature_channels
        self.num_scales = len(feature_channels)
        self.sparsity_lambda = sparsity_lambda
        
        # 为每个尺度创建独立的门控网络
        self.gate_networks = nn.ModuleList()
        self.sparse_gates = nn.ModuleList()
        
        for channels in feature_channels:
            hidden_dim = int(channels * hidden_ratio)
            
            # 每个尺度的门控权重生成网络 (在通道维度上操作)
            gate_network = nn.Sequential(
                # 使用全局平均池化获取通道注意力
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, channels),
                nn.Sigmoid()
            )
            self.gate_networks.append(gate_network)
            
            # 每个尺度的稀疏门控
            sparse_gate = DifferentiableSparseGate(temperature=temperature)
            self.sparse_gates.append(sparse_gate)
        
        # 跨尺度交互模块（轻量级，可选）
        self.cross_scale_interaction = nn.Sequential(
            nn.Linear(sum(feature_channels), sum(feature_channels) // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(sum(feature_channels) // 2, sum(feature_channels)),
            nn.Sigmoid()
        )
        


    # 修改 forward 方法的返回部分
    def forward(self, multi_scale_features, training):
        """
        参数:
            multi_scale_features: 多尺度特征OrderedDict {'1': feat1, '2': feat2, ...}
                feat_i: [batch_size, height, width, channels]
            training: 训练模式标志
        """
        # 修改断言检查，验证输入是否为OrderedDict且长度匹配
        assert isinstance(multi_scale_features, dict) or hasattr(multi_scale_features, 'keys')
        assert len(multi_scale_features) == self.num_scales
        
        batch_size = list(multi_scale_features.values())[0].shape[0]
        # 使用 OrderedDict 存储结果，保持键值一致性
        gated_features = OrderedDict()
        binary_gates = OrderedDict()
        importance_weights = OrderedDict()
        
        # 1. 对各尺度特征分别进行动态选择
        # 修改循环部分，直接遍历 items()
        for i, (scale_key, (features, gate_network, sparse_gate)) in enumerate(
            zip(multi_scale_features.keys(), zip(multi_scale_features.values(), self.gate_networks, self.sparse_gates))):
            
            # features: [B, H, W, C]
            # 转换为 [B, C, H, W] 以适应PyTorch的通道优先格式
            features_transposed = features.permute(0, 3, 1, 2)
            
            # 生成该尺度特征的重要性权重 (在通道维度上)
            scale_importance = gate_network(features_transposed)  # [batch_size, channels]
            
            # 应用稀疏门控
            scale_binary_gate = sparse_gate(scale_importance, training=training)  # [B, C]
            
            # 应用门控得到提纯后的特征 (在通道维度上广播)
            # 调整维度以进行广播: [B, C] -> [B, C, 1, 1]
            gate_expanded = scale_binary_gate.unsqueeze(-1).unsqueeze(-1)
            gated_features_transposed = features_transposed * gate_expanded  # [B, C, H, W]
            
            # 转换回 [B, H, W, C] 格式
            scale_gated_features = gated_features_transposed.permute(0, 2, 3, 1)
            
            # 使用相同的键保存结果
            gated_features[scale_key] = scale_gated_features
            binary_gates[scale_key] = scale_binary_gate
            importance_weights[scale_key] = scale_importance
        
        # 2. 跨尺度交互（可选，增强各尺度间的协调性）
        if self.num_scales > 1:
            # 拼接所有尺度的通道特征进行交互
            all_importance_weights = [importance_weights[key] for key in multi_scale_features.keys()]
            all_channel_features = torch.cat(all_importance_weights, dim=1)  # [batch_size, total_channels]
            
            # 生成跨尺度交互权重
            cross_scale_weights = self.cross_scale_interaction(all_channel_features)
            
            # 按原始尺度分割交互权重并应用
            split_weights = torch.split(cross_scale_weights, self.feature_channels, dim=1)
            
            final_features = OrderedDict()
            for i, (key, feat) in enumerate(gated_features.items()):
                weight = split_weights[i]
                # 使用交互权重对特征进行微调
                # weight: [B, C] -> [B, C, 1, 1] -> [B, H, W, C] (通过广播)
                weight_expanded = weight.unsqueeze(1).unsqueeze(1)
                adjusted_feat = feat * weight_expanded
                final_features[key] = adjusted_feat
            
            gated_features = final_features
        
        return {
            'gated_features': gated_features,      # 提纯后的多尺度特征 OrderedDict
            'binary_gates': binary_gates,          # 各尺度的二值化门控信号 OrderedDict
            'importance_weights': importance_weights  # 各尺度的重要性权重（软）OrderedDict
        }
    
    def compute_sparsity_loss(self, binary_gates):
        """计算多尺度稀疏性损失
        
        参数:
            binary_gates: OrderedDict，包含各尺度的二值化门控信号
                        {'1': gate1, '2': gate2, ...}，每个gate形状为[B, C]
        
        返回:
            平均稀疏性损失值
        """
        sparsity_loss = 0.0
        # 修改：遍历OrderedDict的values而不是直接遍历
        for gate in binary_gates.values():
            # 鼓励门控尽可能稀疏（非零元素比例小）
            sparsity_ratio = gate.mean()  # 均值越接近0，说明越稀疏
            sparsity_loss += sparsity_ratio
        
        # 修改：使用len(binary_gates)确保正确计算平均值
        return sparsity_loss / len(binary_gates)
    



    # 测试示例
if __name__ == "__main__":
    # 模拟多尺度特征图，具有不同的空间尺寸和通道数
    batch_size = 4
    
    # 创建具有指定形状的模拟输入 [B, H, W, C]
    multi_scale_input = [
        torch.randn(batch_size, 128, 228, 256),   # 尺度1
        torch.randn(batch_size, 64, 114, 512),    # 尺度2
    ]
    
    # 提取通道数作为特征维度
    feature_channels = [feat.shape[-1] for feat in multi_scale_input]
    
    print("原始特征形状:", [feat.shape for feat in multi_scale_input])
    print("特征通道数:", feature_channels)
    
    # 创建多尺度特征选择器
    selector = MultiScaleFeatureSelector(feature_channels=feature_channels)
    
    # 前向传播
    outputs = selector(multi_scale_input, training=True)
    
    print("输入形状:", [feat.shape for feat in multi_scale_input])
    print("输出形状:", [feat.shape for feat in outputs['gated_features']])
    print("门控形状:", [gate.shape for gate in outputs['binary_gates']])
    
    # 验证输出形状与输入形状一致
    for i, (input_feat, output_feat) in enumerate(zip(multi_scale_input, outputs['gated_features'])):
        assert input_feat.shape == output_feat.shape, f"尺度{i}的输出形状与输入形状不匹配"
    
    # 计算稀疏性
    for i, gate in enumerate(outputs['binary_gates']):
        sparsity = (gate < 0.1).float().mean()
        print(f"尺度{i+1}稀疏度: {sparsity:.3f}")
    
    # 计算稀疏损失
    sparsity_loss = selector.compute_sparsity_loss(outputs['binary_gates'])
    print(f"总稀疏损失: {sparsity_loss:.4f}")