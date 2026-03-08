import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, 
                 embedding_dim=64, hidden_dims=[64, 32], dropout=0.1):
        super(DeepFM, self).__init__()
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
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        user_feat_emb = self.user_fc(user_features)
        item_feat_emb = self.item_fc(item_features)

        color_emb = self.color_embedding(user_color_idx)
        size_emb = self.size_embedding(user_size_idx)

        combined_emb = torch.cat([user_emb, item_emb, user_feat_emb, item_feat_emb, 
                                color_emb, size_emb], dim=-1)

        combined_emb = self.dropout(combined_emb)

        fm_part = (combined_emb[:, :self.embedding_dim] * combined_emb[:, self.embedding_dim:2*self.embedding_dim]).sum(dim=-1, keepdim=True)

        dnn_part = self.dnn(combined_emb)

        output = fm_part + dnn_part
        return output