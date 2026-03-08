import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEmbedding(nn.Module):
    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, 
                 embedding_dim=64):
        super(FeatureEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.color_embedding = nn.Embedding(22, embedding_dim // 2)
        self.size_embedding = nn.Embedding(18, embedding_dim // 2)
        
        self.user_feature_proj = nn.Linear(user_feature_dim, embedding_dim)
        self.item_feature_proj = nn.Linear(item_feature_dim, embedding_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.color_embedding.weight)
        nn.init.xavier_uniform_(self.size_embedding.weight)
        
    def forward(self, user_idx, user_features, user_color_idx, user_size_idx, 
                item_idx, item_features):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        color_emb = self.color_embedding(user_color_idx)
        size_emb = self.size_embedding(user_size_idx)
        
        user_feat_proj = self.user_feature_proj(user_features)
        item_feat_proj = self.item_feature_proj(item_features)
        
        user_features_combined = torch.cat([user_emb, color_emb, size_emb, user_feat_proj], dim=-1)
        item_features_combined = torch.cat([item_emb, item_feat_proj], dim=-1)
        
        return user_features_combined, item_features_combined

class CrossAttention(nn.Module):
    def __init__(self, user_dim, item_dim, output_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        self.user_proj = nn.Linear(user_dim, output_dim)
        self.item_proj = nn.Linear(item_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        self.scale = (output_dim // num_heads) ** -0.5
        
    def forward(self, user_features, item_features):
        batch_size = user_features.shape[0]
        
        Q = self.user_proj(user_features)
        K = self.item_proj(item_features)
        V = self.item_proj(item_features)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.output_dim // self.num_heads)
        K = K.view(batch_size, -1, self.num_heads, self.output_dim // self.num_heads)
        V = V.view(batch_size, -1, self.num_heads, self.output_dim // self.num_heads)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.output_dim)
        out = self.out_proj(out)
        
        return out.squeeze(1)

class FeatureInteraction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureInteraction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x + residual
        return x

class MultiScaleAttention(nn.Module):
    def __init__(self, embed_dim, num_heads_list=[2, 4, 8]):
        super(MultiScaleAttention, self).__init__()
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for num_heads in num_heads_list
        ])
        self.fusion = nn.Linear(embed_dim * len(num_heads_list), embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        outputs = []
        
        for attn in self.attention_layers:
            out, _ = attn(x, x, x)
            outputs.append(out.squeeze(1))
        
        concatenated = torch.cat(outputs, dim=-1)
        fused = self.fusion(concatenated)
        output = self.norm(fused + x.squeeze(1))
        
        return output

class HybridAttentionRecommendationNetwork(nn.Module):
    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, 
                 embedding_dim=64, hidden_dims=[256, 128, 64], dropout=0.2):
        super(HybridAttentionRecommendationNetwork, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        self.feature_embedding = FeatureEmbedding(
            num_users, num_items, user_feature_dim, item_feature_dim, embedding_dim
        )
        
        user_combined_dim = embedding_dim + embedding_dim // 2 + embedding_dim // 2 + embedding_dim
        item_combined_dim = embedding_dim + embedding_dim
        
        self.cross_attention = CrossAttention(
            user_combined_dim, item_combined_dim, embedding_dim, num_heads=4
        )
        
        self.multi_scale_attention = MultiScaleAttention(embedding_dim, num_heads_list=[2, 4, 8])
        
        self.feature_interaction1 = FeatureInteraction(embedding_dim, hidden_dims[0])
        self.feature_interaction2 = FeatureInteraction(embedding_dim, hidden_dims[1])
        
        self.prediction_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], 1)
        )
        
        self.output_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, user_idx, user_features, user_color_idx, user_size_idx, 
                item_idx, item_features):
        user_combined, item_combined = self.feature_embedding(
            user_idx, user_features, user_color_idx, user_size_idx, 
            item_idx, item_features
        )
        
        cross_attended = self.cross_attention(user_combined, item_combined)
        
        multi_scale_attended = self.multi_scale_attention(cross_attended)
        
        interacted1 = self.feature_interaction1(multi_scale_attended)
        interacted2 = self.feature_interaction2(interacted1)
        
        normalized = self.output_norm(interacted2)
        
        score = self.prediction_network(normalized)
        
        return score
    
    def get_user_item_interaction(self, user_idx, user_features, user_color_idx, 
                                  user_size_idx, item_idx, item_features):
        user_combined, item_combined = self.feature_embedding(
            user_idx, user_features, user_color_idx, user_size_idx, 
            item_idx, item_features
        )
        
        cross_attended = self.cross_attention(user_combined, item_combined)
        multi_scale_attended = self.multi_scale_attention(cross_attended)
        interacted1 = self.feature_interaction1(multi_scale_attended)
        interacted2 = self.feature_interaction2(interacted1)
        normalized = self.output_norm(interacted2)
        
        return normalized