import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        self.scale = self.scale.to(query.device)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        output = self.out_linear(x)
        return output.squeeze(1)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class FeatureFusionLayer(nn.Module):
    def __init__(self, feature_dims, output_dim):
        super(FeatureFusionLayer, self).__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        self.feature_projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        
        self.attention_weights = nn.Sequential(
            nn.Linear(output_dim * len(feature_dims), output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, len(feature_dims)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features):
        projected_features = []
        for i, (feat, proj) in enumerate(zip(features, self.feature_projections)):
            projected_features.append(proj(feat))
        
        concatenated = torch.cat(projected_features, dim=-1)
        attention_weights = self.attention_weights(concatenated)
        
        fused = sum(w.unsqueeze(-1) * feat for w, feat in zip(attention_weights.unbind(-1), projected_features))
        return fused

class OptimizedTwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, 
                 embedding_dim=128, hidden_dims=[256, 128, 64], dropout=0.2, 
                 num_heads=4, temperature=0.07):
        super(OptimizedTwoTowerModel, self).__init__()
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
        
        user_feature_dims = [user_feature_dim, embedding_dim, embedding_dim, embedding_dim]
        self.user_feature_fusion = FeatureFusionLayer(user_feature_dims, embedding_dim)
        
        item_feature_dims = [item_feature_dim, embedding_dim]
        self.item_feature_fusion = FeatureFusionLayer(item_feature_dims, embedding_dim)
        
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dims[0], hidden_dims[1], dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dims[1], hidden_dims[2], dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], embedding_dim)
        )
        
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dims[0], hidden_dims[1], dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dims[1], hidden_dims[2], dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], embedding_dim)
        )
        
        self.user_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.item_attention = MultiHeadAttention(embedding_dim, num_heads)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def user_forward(self, user_idx, user_features, user_color_idx, user_size_idx):
        user_emb = self.user_embedding(user_idx)
        color_emb = self.color_embedding(user_color_idx)
        size_emb = self.size_embedding(user_size_idx)
        
        fused_features = self.user_feature_fusion([user_features, user_emb, color_emb, size_emb])
        
        user_vector = self.user_tower(fused_features)
        
        user_vector = self.user_attention(user_vector.unsqueeze(1), user_vector.unsqueeze(1), user_vector.unsqueeze(1))
        user_vector = self.layer_norm(user_vector)
        
        return user_vector
    
    def item_forward(self, item_idx, item_features):
        item_emb = self.item_embedding(item_idx)
        
        fused_features = self.item_feature_fusion([item_features, item_emb])
        
        item_vector = self.item_tower(fused_features)
        
        item_vector = self.item_attention(item_vector.unsqueeze(1), item_vector.unsqueeze(1), item_vector.unsqueeze(1))
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