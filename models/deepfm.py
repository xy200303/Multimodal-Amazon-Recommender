"""DeepFM 推荐模型。

该实现将 ID 嵌入、用户侧统计特征、商品侧特征以及颜色/尺码偏好共同输入，
同时保留 FM 的低阶交互与 DNN 的高阶非线性表达能力。
"""

import torch
import torch.nn as nn

class DeepFM(nn.Module):
    """融合显式特征与隐式嵌入的 DeepFM 排序模型。"""
    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, 
                 embedding_dim=64, hidden_dims=[64, 32], dropout=0.1,
                 num_colors=23, num_sizes=19):
        super(DeepFM, self).__init__()
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

        self.user_fc = nn.Linear(user_feature_dim, embedding_dim)
        self.item_fc = nn.Linear(item_feature_dim, embedding_dim)

        nn.init.xavier_uniform_(self.user_fc.weight)
        nn.init.xavier_uniform_(self.item_fc.weight)

        self.dnn = nn.Sequential(
            nn.Linear(embedding_dim * 6, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, user_idx, user_features, user_color_idx, user_size_idx, item_idx, item_features):
        """输出用户与目标商品之间的二分类 logits 分数。"""
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        user_feat_emb = self.user_fc(user_features)
        item_feat_emb = self.item_fc(item_features)

        color_emb = self.color_embedding(user_color_idx)
        size_emb = self.size_embedding(user_size_idx)

        combined_emb = torch.cat([user_emb, item_emb, user_feat_emb, item_feat_emb, 
                                color_emb, size_emb], dim=-1)

        combined_emb = self.dropout(combined_emb)

        # FM 部分显式建模用户 ID 与商品 ID 的二阶交互。
        fm_part = (combined_emb[:, :self.embedding_dim] * combined_emb[:, self.embedding_dim:2*self.embedding_dim]).sum(dim=-1, keepdim=True)

        # DNN 部分学习高阶非线性交互模式。
        dnn_part = self.dnn(combined_emb)

        output = fm_part + dnn_part
        return output
