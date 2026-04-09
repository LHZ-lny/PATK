import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightScalingNetwork(nn.Module):
    """轻量化伸缩权重网络"""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出在 [0, 1] 范围内
        )

    def forward(self, x):
        return self.network(x)


class DifferentiableTimeScaling(nn.Module):
    """可微时序伸缩模块"""

    def __init__(self, patching_length, features, hidden_dim=32):
        super().__init__()
        self.patching_length = patching_length
        self.features = features

        # 轻量化伸缩网络
        self.scaling_network = LightweightScalingNetwork(
            input_dim=features * 2,  # 均值和标准差
            hidden_dim=hidden_dim
        )

    def forward(self, patches):
        """
        参数:
        - patches: 输入子序列 [batch_size, patching_num, patching_length, features]

        返回:
        - scaled_patches: 伸缩后的子序列 [batch_size, patching_num, patching_length, features]
        - scaling_weights: 伸缩权重 [batch_size, patching_num, 1]
        """
        batch_size, patching_num, patching_length, features = patches.shape

        # 1. 计算伸缩权重
        scaling_weights = self._compute_scaling_weights(patches)

        # 2. 将权重映射到伸缩因子 [0.5, 2.0]
        scale_factors = 0.5 + scaling_weights * 1.5

        # 3. 应用可微伸缩
        scaled_patches = self._apply_scaling(patches, scale_factors)

        return scaled_patches, scaling_weights

    def _compute_scaling_weights(self, patches):
        """计算每个子序列的伸缩权重"""
        batch_size, patching_num, patching_length, features = patches.shape

        # 重塑以便处理 [batch_size * patching_num, patching_length, features]
        patches_flat = patches.reshape(batch_size * patching_num, patching_length, features)

        # 计算统计特征
        mean_features = patches_flat.mean(dim=1)  # [batch_size * patching_num, features]
        std_features = patches_flat.std(dim=1)  # [batch_size * patching_num, features]

        # 组合特征作为网络输入
        network_input = torch.cat([mean_features, std_features], dim=1)

        # 计算权重 [batch_size * patching_num, 1]
        scaling_weights = self.scaling_network(network_input)

        # 重塑为原始形状 [batch_size, patching_num, 1]
        scaling_weights = scaling_weights.reshape(batch_size, patching_num, 1)

        return scaling_weights

    def _apply_scaling(self, patches, scale_factors):
        """应用可微伸缩变换"""
        batch_size, patching_num, patching_length, features = patches.shape

        # 重塑以便批处理 [batch_size * patching_num, features, patching_length]
        patches_transposed = patches.reshape(
            batch_size * patching_num, patching_length, features
        ).transpose(1, 2)

        # 生成归一化的索引网格 [-1, 1]
        indices = self._create_scaling_indices(
            patching_length, scale_factors, batch_size * patching_num
        )

        # 使用grid_sample进行可微重采样
        # 添加通道维度 [batch_size * patching_num, 1, features, patching_length]
        patches_with_channel = patches_transposed.unsqueeze(1)

        # 使用双线性插值进行重采样
        scaled_patches = F.grid_sample(
            patches_with_channel,
            indices,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze(1)  # [batch_size * patching_num, features, patching_length]

        # 恢复原始形状 [batch_size, patching_num, patching_length, features]
        scaled_patches = scaled_patches.transpose(1, 2).reshape(
            batch_size, patching_num, patching_length, features
        )

        return scaled_patches

    def _create_scaling_indices(self, length, scale_factors, batch_size):
        """创建用于伸缩的归一化索引网格"""
        device = scale_factors.device

        # 创建基础网格 [-1, 1]
        base_grid = torch.linspace(-1, 1, length, device=device)  # [length]

        # 应用伸缩因子 [batch_size, 1] -> [batch_size, length]
        scale_factors_flat = scale_factors.reshape(batch_size, 1)
        scaled_grid = base_grid.unsqueeze(0) / scale_factors_flat

        # 限制在 [-1, 1] 范围内
        scaled_grid = torch.clamp(scaled_grid, -1, 1)

        # 创建特征维度的网格 (保持不变)
        feature_grid = torch.zeros_like(scaled_grid)  # [batch_size, length]

        # 组合成grid_sample所需的格式 [batch_size, length, 2]
        # 第一个通道是特征维度，第二个通道是时间维度
        indices = torch.stack([feature_grid, scaled_grid], dim=-1)

        return indices



# 初始化
scaling_module = DifferentiableTimeScaling(
    patching_length=10,  # 子序列长度
    features=3           # 特征维度
)

# 输入: [batch_size, patching_num, patching_length, features]
patches = torch.randn(32, 5, 10, 3)

# 应用伸缩
scaled_patches, scaling_weights = scaling_module(patches)

print(f"输入形状: {patches.shape}")
print(f"输出形状: {scaled_patches.shape}")
print(f"伸缩权重: {scaling_weights.shape}")

# 验证可微性
loss = scaled_patches.sum()
loss.backward()