import torch
import torch.nn as nn
import numpy as np

class CollaborativeFiltering(nn.Module):
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
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        user_emb = self.dropout(user_emb)
        item_emb = self.dropout(item_emb)

        score = (user_emb * item_emb).sum(dim=-1, keepdim=True)
        return score