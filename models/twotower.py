"""基础多模态双塔推荐模型。

用户塔融合用户画像、颜色偏好、尺码偏好与 ID 嵌入；
商品塔融合商品结构化特征与商品 ID 嵌入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    """本文核心双塔模型的基础版本。"""
    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, 
                 embedding_dim=64, hidden_dims=[128, 64], dropout=0.3,
                 num_colors=23, num_sizes=19):
        super(TwoTowerModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.color_embedding = nn.Embedding(num_colors, embedding_dim)
        self.size_embedding = nn.Embedding(num_sizes, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.color_embedding.weight)
        nn.init.xavier_uniform_(self.size_embedding.weight)

        # 先把用户/商品显式特征投影到统一维度，再与 ID 嵌入拼接。
        self.user_feature_proj = nn.Sequential(
            nn.Linear(user_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.item_feature_proj = nn.Sequential(
            nn.Linear(item_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], embedding_dim)
        )

        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], embedding_dim)
        )

        self.user_norm = nn.LayerNorm(embedding_dim)
        self.item_norm = nn.LayerNorm(embedding_dim)
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def user_forward(self, user_idx, user_features, user_color_idx, user_size_idx):
        """生成归一化的用户塔向量。"""
        user_emb = self.user_embedding(user_idx)
        color_emb = self.color_embedding(user_color_idx)
        size_emb = self.size_embedding(user_size_idx)
        
        user_feat_proj = self.user_feature_proj(user_features)
        
        user_input = torch.cat([user_feat_proj, user_emb, color_emb, size_emb], dim=-1)
        user_vector = self.user_tower(user_input)
        user_vector = self.user_norm(user_vector)
        
        return F.normalize(user_vector, p=2, dim=-1)

    def item_forward(self, item_idx, item_features):
        """生成归一化的商品塔向量。"""
        item_emb = self.item_embedding(item_idx)
        
        item_feat_proj = self.item_feature_proj(item_features)
        
        item_input = torch.cat([item_feat_proj, item_emb], dim=-1)
        item_vector = self.item_tower(item_input)
        item_vector = self.item_norm(item_vector)
        
        return F.normalize(item_vector, p=2, dim=-1)

    def forward(self, user_idx, user_features, user_color_idx, user_size_idx, item_idx, item_features):
        """返回用户向量与商品向量的缩放点积分数。"""
        user_vector = self.user_forward(user_idx, user_features, user_color_idx, user_size_idx)
        item_vector = self.item_forward(item_idx, item_features)
        
        score = torch.sum(user_vector * item_vector, dim=-1, keepdim=True) / self.temperature.abs()
        
        return score
