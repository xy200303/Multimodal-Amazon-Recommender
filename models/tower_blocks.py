"""双塔模型共享的轻量基础模块。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """将多路输入先投影到统一维度，再用门控方式融合。"""

    def __init__(self, feature_dims, output_dim):
        super().__init__()
        self.feature_projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        self.gate = nn.Sequential(
            nn.Linear(output_dim * len(feature_dims), output_dim),
            nn.Sigmoid()
        )

    def forward(self, features):
        projected_features = []
        for feat, projection in zip(features, self.feature_projections):
            projected_features.append(projection(feat))

        gate_weights = self.gate(torch.cat(projected_features, dim=-1))
        fused = sum(
            weight.unsqueeze(-1) * feature
            for weight, feature in zip(gate_weights.unbind(-1), projected_features)
        )
        return fused


class EfficientBlock(nn.Module):
    """带残差连接的轻量前馈块。"""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm2(x)
        return F.gelu(x)
