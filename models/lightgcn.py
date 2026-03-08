import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class LightGCNConv(MessagePassing):
    def __init__(self):
        super(LightGCNConv, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class LightGCN(nn.Module):
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
        user_emb, item_emb = self.forward(edge_index, user_features, item_features)
        user_emb = user_emb[user_ids]
        item_emb = item_emb[item_ids]
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores

class MultiModalLightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3,
                 user_numeric_dim=12, user_vector_dim=3, num_colors=22, 
                 num_sizes=18, item_numeric_dim=5, item_vector_dim=12,
                 dropout=0.1):
        super(MultiModalLightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

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
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        if user_features is not None:
            user_numeric_emb = self.user_numeric_fc(user_features[:, :12])
            user_vector_emb = self.user_vector_fc(user_features[:, 12:15])
            user_emb = user_emb + user_numeric_emb + user_vector_emb

        if user_color_indices is not None:
            color_emb = self.color_embedding(user_color_indices)
            user_emb = user_emb + color_emb

        if user_size_indices is not None:
            size_emb = self.size_embedding(user_size_indices)
            user_emb = user_emb + size_emb

        if item_features is not None:
            item_numeric_emb = self.item_numeric_fc(item_features[:, :5])
            item_vector_emb = self.item_vector_fc(item_features[:, 5:])
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
        user_emb, item_emb = self.forward(edge_index, user_features, user_color_indices,
                                          user_size_indices, item_features)
        user_emb = user_emb[user_ids]
        item_emb = item_emb[item_ids]
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores