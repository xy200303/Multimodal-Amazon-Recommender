import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedTwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, 
                 embedding_dim=64, hidden_dims=[128, 64], dropout=0.2, 
                 temperature=0.07):
        super(SimplifiedTwoTowerModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.color_embedding = nn.Embedding(22, embedding_dim)
        self.size_embedding = nn.Embedding(18, embedding_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.color_embedding.weight)
        nn.init.xavier_uniform_(self.size_embedding.weight)
        
        user_input_dim = user_feature_dim + embedding_dim * 3
        item_input_dim = item_feature_dim + embedding_dim
        
        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, hidden_dims[0]),
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
            nn.Linear(item_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], embedding_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def user_forward(self, user_idx, user_features, user_color_idx, user_size_idx):
        user_emb = self.user_embedding(user_idx)
        color_emb = self.color_embedding(user_color_idx)
        size_emb = self.size_embedding(user_size_idx)
        
        user_input = torch.cat([user_features, user_emb, color_emb, size_emb], dim=-1)
        user_vector = self.user_tower(user_input)
        user_vector = self.layer_norm(user_vector)
        
        return user_vector
    
    def item_forward(self, item_idx, item_features):
        item_emb = self.item_embedding(item_idx)
        item_input = torch.cat([item_features, item_emb], dim=-1)
        item_vector = self.item_tower(item_input)
        item_vector = self.layer_norm(item_vector)
        
        return item_vector
    
    def forward(self, user_idx, user_features, user_color_idx, user_size_idx, item_idx, item_features):
        user_vector = self.user_forward(user_idx, user_features, user_color_idx, user_size_idx)
        item_vector = self.item_forward(item_idx, item_features)
        
        normalized_user = F.normalize(user_vector, p=2, dim=-1)
        normalized_item = F.normalize(item_vector, p=2, dim=-1)
        
        score = torch.sum(normalized_user * normalized_item, dim=-1, keepdim=True) / self.temperature
        
        return score
    
    def get_user_embeddings(self, user_idx, user_features, user_color_idx, user_size_idx):
        return self.user_forward(user_idx, user_features, user_color_idx, user_size_idx)
    
    def get_item_embeddings(self, item_idx, item_features):
        return self.item_forward(item_idx, item_features)