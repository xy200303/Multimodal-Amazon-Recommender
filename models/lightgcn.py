"""LightGCN 及其多模态扩展实现。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

def build_lightgcn_norm(edge_index, num_nodes, dtype):
    """Precompute symmetric normalization weights for a fixed graph."""
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    return deg_inv_sqrt[row] * deg_inv_sqrt[col]

class LightGCNConv(MessagePassing):
    """LightGCN 的图卷积传播层。"""
    def __init__(self):
        super(LightGCNConv, self).__init__(aggr='add')

    def forward(self, x, edge_index, norm=None):
        """按 LightGCN 的归一化规则进行消息传播。"""
        if norm is None:
            norm = build_lightgcn_norm(edge_index, x.size(0), x.dtype)
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class LightGCN(nn.Module):
    """标准 LightGCN 模型。"""
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3,
                 user_feat_dim=0, item_feat_dim=0, dropout=0.1):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

        if user_feat_dim > 0:
            self.user_feat_transform = nn.Linear(user_feat_dim, embedding_dim)
        
        if item_feat_dim > 0:
            self.item_feat_transform = nn.Linear(item_feat_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index, user_features=None, item_features=None):
        """在用户-商品图上进行多层传播，输出最终嵌入。"""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        if user_features is not None and self.user_feat_dim > 0:
            user_feat_emb = self.user_feat_transform(user_features)
            user_emb = user_emb + user_feat_emb

        if item_features is not None and self.item_feat_dim > 0:
            item_feat_emb = self.item_feat_transform(item_features)
            item_emb = item_emb + item_feat_emb

        all_embeddings = torch.cat([user_emb, item_emb], dim=0)

        embeddings_list = [all_embeddings]

        for conv in self.convs:
            all_embeddings = conv(all_embeddings, edge_index)
            all_embeddings = self.dropout(all_embeddings)
            embeddings_list.append(all_embeddings)

        final_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)

        user_final_emb = final_embeddings[:self.num_users]
        item_final_emb = final_embeddings[self.num_users:]

        return user_final_emb, item_final_emb

    def predict(self, user_ids, item_ids, edge_index, user_features=None, item_features=None):
        """根据传播后的嵌入计算指定用户和商品的匹配分数。"""
        user_emb, item_emb = self.forward(edge_index, user_features, item_features)
        user_emb = user_emb[user_ids]
        item_emb = item_emb[item_ids]
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores

class MultiModalLightGCN(nn.Module):
    """融合用户/商品多模态特征的 LightGCN 版本。"""
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3,
                 user_numeric_dim=12, user_vector_dim=3, num_colors=23, 
                 num_sizes=19, item_numeric_dim=5, item_vector_dim=12,
                 dropout=0.1):
        super(MultiModalLightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_numeric_dim = user_numeric_dim
        self.user_vector_dim = user_vector_dim
        self.item_numeric_dim = item_numeric_dim
        self.item_vector_dim = item_vector_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.user_numeric_fc = nn.Linear(user_numeric_dim, embedding_dim)
        self.user_vector_fc = nn.Linear(user_vector_dim, embedding_dim)
        self.color_embedding = nn.Embedding(num_colors, embedding_dim)
        self.size_embedding = nn.Embedding(num_sizes, embedding_dim)
        self.item_numeric_fc = nn.Linear(item_numeric_dim, embedding_dim)
        self.item_vector_fc = nn.Linear(item_vector_dim, embedding_dim)

        nn.init.xavier_uniform_(self.user_numeric_fc.weight)
        nn.init.xavier_uniform_(self.user_vector_fc.weight)
        nn.init.xavier_uniform_(self.color_embedding.weight)
        nn.init.xavier_uniform_(self.size_embedding.weight)
        nn.init.xavier_uniform_(self.item_numeric_fc.weight)
        nn.init.xavier_uniform_(self.item_vector_fc.weight)

        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index, user_features=None, user_color_indices=None, 
                user_size_indices=None, item_features=None):
        """将显式特征注入初始嵌入后进行图传播。"""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        if user_features is not None:
            user_numeric_emb = self.user_numeric_fc(user_features[:, :self.user_numeric_dim])
            user_vector_emb = self.user_vector_fc(
                user_features[:, self.user_numeric_dim:self.user_numeric_dim + self.user_vector_dim]
            )
            user_emb = user_emb + user_numeric_emb + user_vector_emb

        if user_color_indices is not None:
            color_emb = self.color_embedding(user_color_indices)
            user_emb = user_emb + color_emb

        if user_size_indices is not None:
            size_emb = self.size_embedding(user_size_indices)
            user_emb = user_emb + size_emb

        if item_features is not None:
            item_numeric_emb = self.item_numeric_fc(item_features[:, :self.item_numeric_dim])
            item_vector_emb = self.item_vector_fc(
                item_features[:, self.item_numeric_dim:self.item_numeric_dim + self.item_vector_dim]
            )
            item_emb = item_emb + item_numeric_emb + item_vector_emb

        all_embeddings = torch.cat([user_emb, item_emb], dim=0)

        embeddings_list = [all_embeddings]

        for conv in self.convs:
            all_embeddings = conv(all_embeddings, edge_index)
            all_embeddings = self.dropout(all_embeddings)
            embeddings_list.append(all_embeddings)

        final_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)

        user_final_emb = final_embeddings[:self.num_users]
        item_final_emb = final_embeddings[self.num_users:]

        return user_final_emb, item_final_emb

    def predict(self, user_ids, item_ids, edge_index, user_features=None,
                user_color_indices=None, user_size_indices=None, item_features=None):
        """输出指定用户与指定商品的匹配分数。"""
        user_emb, item_emb = self.forward(edge_index, user_features, user_color_indices,
                                          user_size_indices, item_features)
        user_emb = user_emb[user_ids]
        item_emb = item_emb[item_ids]
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores
