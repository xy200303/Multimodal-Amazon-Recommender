import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, 
                 embedding_dim=128, hidden_dims=[256, 128], dropout=0.2):
        super(TwoTowerModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.color_embedding = nn.Embedding(22, embedding_dim)
        self.size_embedding = nn.Embedding(18, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.color_embedding.weight)
        nn.init.xavier_uniform_(self.size_embedding.weight)

        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim + embedding_dim * 3, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], embedding_dim)
        )

        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_dim + embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], embedding_dim)
        )

    def user_forward(self, user_idx, user_features, user_color_idx, user_size_idx):
        user_emb = self.user_embedding(user_idx)
        color_emb = self.color_embedding(user_color_idx)
        size_emb = self.size_embedding(user_size_idx)
        
        user_input = torch.cat([user_features, user_emb, color_emb, size_emb], dim=-1)
        user_vector = self.user_tower(user_input)
        
        return user_vector

    def item_forward(self, item_idx, item_features):
        item_emb = self.item_embedding(item_idx)
        item_input = torch.cat([item_features, item_emb], dim=-1)
        item_vector = self.item_tower(item_input)
        
        return item_vector

    def forward(self, user_idx, user_features, user_color_idx, user_size_idx, item_idx, item_features):
        user_vector = self.user_forward(user_idx, user_features, user_color_idx, user_size_idx)
        item_vector = self.item_forward(item_idx, item_features)
        
        score = (user_vector * item_vector).sum(dim=-1, keepdim=True)
        return score