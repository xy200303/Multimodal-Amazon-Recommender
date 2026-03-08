import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, feature_dims, output_dim):
        super(GatedFusion, self).__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        self.feature_projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        
        self.gate = nn.Sequential(
            nn.Linear(output_dim * len(feature_dims), output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        projected_features = []
        for feat, proj in zip(features, self.feature_projections):
            projected_features.append(proj(feat))
        
        concatenated = torch.cat(projected_features, dim=-1)
        gate_weights = self.gate(concatenated)
        
        fused = sum(w.unsqueeze(-1) * feat for w, feat in zip(gate_weights.unbind(-1), projected_features))
        return fused

class SimplifiedAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SimplifiedAttention, self).__init__()
        self.embed_dim = embed_dim
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        self.scale = embed_dim ** -0.5
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, V)
        out = self.out(out)
        
        return out.squeeze(1)

class EfficientBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(EfficientBlock, self).__init__()
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
        x = F.gelu(x)
        
        return x

class EfficientTwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, 
                 embedding_dim=96, hidden_dims=[192, 96], dropout=0.1, 
                 temperature=0.07):
        super(EfficientTwoTowerModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.color_embedding = nn.Embedding(22, embedding_dim // 2)
        self.size_embedding = nn.Embedding(18, embedding_dim // 2)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.color_embedding.weight)
        nn.init.xavier_uniform_(self.size_embedding.weight)
        
        user_feature_dims = [user_feature_dim, embedding_dim, embedding_dim // 2, embedding_dim // 2]
        self.user_fusion = GatedFusion(user_feature_dims, embedding_dim)
        
        item_feature_dims = [item_feature_dim, embedding_dim]
        self.item_fusion = GatedFusion(item_feature_dims, embedding_dim)
        
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            EfficientBlock(hidden_dims[0], hidden_dims[1], dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], embedding_dim)
        )
        
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            EfficientBlock(hidden_dims[0], hidden_dims[1], dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], embedding_dim)
        )
        
        self.user_attention = SimplifiedAttention(embedding_dim)
        self.item_attention = SimplifiedAttention(embedding_dim)
        
        self.output_norm = nn.LayerNorm(embedding_dim)
        
    def user_forward(self, user_idx, user_features, user_color_idx, user_size_idx):
        user_emb = self.user_embedding(user_idx)
        color_emb = self.color_embedding(user_color_idx)
        size_emb = self.size_embedding(user_size_idx)
        
        fused = self.user_fusion([user_features, user_emb, color_emb, size_emb])
        
        user_vector = self.user_tower(fused)
        
        user_vector = self.user_attention(user_vector.unsqueeze(1))
        user_vector = self.output_norm(user_vector)
        
        return user_vector
    
    def item_forward(self, item_idx, item_features):
        item_emb = self.item_embedding(item_idx)
        
        fused = self.item_fusion([item_features, item_emb])
        
        item_vector = self.item_tower(fused)
        
        item_vector = self.item_attention(item_vector.unsqueeze(1))
        item_vector = self.output_norm(item_vector)
        
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