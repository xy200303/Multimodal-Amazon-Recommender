"""协同过滤基线模型。

该模型仅使用用户 ID 与商品 ID 的可学习嵌入，通过向量点积得到偏好分数。
适合作为本文复杂多模态模型的对比基线。
"""

import torch
import torch.nn as nn

class CollaborativeFiltering(nn.Module):
    """最基础的隐式反馈协同过滤模型。"""
    def __init__(self, num_users, num_items, embedding_dim=64, dropout=0.1):
        super(CollaborativeFiltering, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, user_idx, item_idx):
        """根据用户嵌入与商品嵌入的点积输出匹配分数。"""
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        user_emb = self.dropout(user_emb)
        item_emb = self.dropout(item_emb)

        score = (user_emb * item_emb).sum(dim=-1, keepdim=True)
        return score
