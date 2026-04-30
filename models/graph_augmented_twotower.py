"""Graph-augmented dual-tower recommender."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lightgcn import LightGCNConv, build_lightgcn_norm
from .tower_blocks import EfficientBlock, GatedFusion


class GraphAugmentedTwoTowerModel(nn.Module):
    """Fuse multimodal content features with lightweight graph enhancement."""

    def __init__(
        self,
        num_users,
        num_items,
        user_feature_dim,
        item_feature_dim,
        embedding_dim=96,
        hidden_dims=None,
        dropout=0.1,
        temperature=0.07,
        num_colors=23,
        num_sizes=19,
        num_graph_layers=1
    ):
        super().__init__()
        hidden_dims = hidden_dims or [192, 96]

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.temperature = float(temperature)
        self.num_graph_layers = num_graph_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.color_embedding = nn.Embedding(num_colors, embedding_dim // 2)
        self.size_embedding = nn.Embedding(num_sizes, embedding_dim // 2)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.color_embedding.weight)
        nn.init.xavier_uniform_(self.size_embedding.weight)

        user_feature_dims = [user_feature_dim, embedding_dim, embedding_dim // 2, embedding_dim // 2]
        item_feature_dims = [item_feature_dim, embedding_dim]
        self.user_fusion = GatedFusion(user_feature_dims, embedding_dim)
        self.item_fusion = GatedFusion(item_feature_dims, embedding_dim)

        self.graph_convs = nn.ModuleList([LightGCNConv() for _ in range(num_graph_layers)])

        self.user_graph_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        self.item_graph_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )

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

        self.output_norm = nn.LayerNorm(embedding_dim)
        self.cached_user_graph = None
        self.cached_item_graph = None
        self.graph_edge_index = None
        self.graph_norm = None

        self.register_buffer('all_user_indices', torch.arange(num_users, dtype=torch.long), persistent=False)
        self.register_buffer('all_item_indices', torch.arange(num_items, dtype=torch.long), persistent=False)

    def _build_user_content(self, user_idx, user_features, user_color_idx, user_size_idx):
        """Build user content features for tower encoding and graph initialization."""
        user_emb = self.user_embedding(user_idx)
        color_emb = self.color_embedding(user_color_idx)
        size_emb = self.size_embedding(user_size_idx)
        return self.user_fusion([user_features, user_emb, color_emb, size_emb])

    def _build_item_content(self, item_idx, item_features):
        """Build item content features for tower encoding and graph initialization."""
        item_emb = self.item_embedding(item_idx)
        return self.item_fusion([item_features, item_emb])

    def _prepare_graph_norm(self, edge_index, dtype):
        """Cache LightGCN normalization weights because the graph is fixed."""
        needs_refresh = (
            self.graph_edge_index is None
            or self.graph_norm is None
            or self.graph_edge_index.shape != edge_index.shape
            or self.graph_edge_index.device != edge_index.device
            or not torch.equal(self.graph_edge_index, edge_index)
        )

        if needs_refresh:
            self.graph_edge_index = edge_index.detach().clone()
            self.graph_norm = build_lightgcn_norm(
                edge_index,
                self.num_users + self.num_items,
                dtype
            )
        elif self.graph_norm.dtype != dtype:
            self.graph_norm = self.graph_norm.to(dtype=dtype)

        return self.graph_norm

    def compute_graph_embeddings(
        self,
        edge_index,
        all_user_features,
        all_user_color_indices,
        all_user_size_indices,
        all_item_features
    ):
        """Run full-graph propagation once and return graph-enhanced user/item embeddings."""
        all_user_indices = self.all_user_indices.to(edge_index.device)
        all_item_indices = self.all_item_indices.to(edge_index.device)

        user_content = self._build_user_content(
            all_user_indices,
            all_user_features,
            all_user_color_indices,
            all_user_size_indices
        )
        item_content = self._build_item_content(all_item_indices, all_item_features)

        all_embeddings = torch.cat([
            F.normalize(user_content, p=2, dim=-1),
            F.normalize(item_content, p=2, dim=-1)
        ], dim=0)

        graph_norm = self._prepare_graph_norm(edge_index, all_embeddings.dtype)
        embedding_history = [all_embeddings]
        propagated = all_embeddings
        for conv in self.graph_convs:
            propagated = conv(propagated, edge_index, graph_norm)
            embedding_history.append(propagated)

        final_embeddings = torch.stack(embedding_history, dim=0).mean(dim=0)
        user_graph = final_embeddings[:self.num_users]
        item_graph = final_embeddings[self.num_users:]
        return F.normalize(user_graph, p=2, dim=-1), F.normalize(item_graph, p=2, dim=-1)

    def refresh_graph_cache(
        self,
        edge_index,
        all_user_features,
        all_user_color_indices,
        all_user_size_indices,
        all_item_features
    ):
        """Refresh graph caches once per epoch or evaluation phase."""
        with torch.inference_mode():
            user_graph, item_graph = self.compute_graph_embeddings(
                edge_index,
                all_user_features,
                all_user_color_indices,
                all_user_size_indices,
                all_item_features
            )
        self.cached_user_graph = user_graph.detach()
        self.cached_item_graph = item_graph.detach()

    def clear_graph_cache(self):
        """Drop stale graph caches after evaluation."""
        self.cached_user_graph = None
        self.cached_item_graph = None

    def _get_cached_user_graph(self, user_idx, fallback_tensor):
        """Read cached user graph features, or return zeros if cache is unavailable."""
        if self.cached_user_graph is None:
            return torch.zeros_like(fallback_tensor)
        return self.cached_user_graph[user_idx]

    def _get_cached_item_graph(self, item_idx, fallback_tensor):
        """Read cached item graph features, or return zeros if cache is unavailable."""
        if self.cached_item_graph is None:
            return torch.zeros_like(fallback_tensor)
        return self.cached_item_graph[item_idx]

    def user_forward(self, user_idx, user_features, user_color_idx, user_size_idx):
        """Encode users by fusing content representation and cached graph representation."""
        user_content = self._build_user_content(user_idx, user_features, user_color_idx, user_size_idx)
        user_graph = self._get_cached_user_graph(user_idx, user_content)
        gate = self.user_graph_gate(torch.cat([user_content, user_graph], dim=-1))
        fused = gate * user_content + (1.0 - gate) * user_graph

        user_vector = self.user_tower(fused)
        user_vector = self.output_norm(user_vector)
        return user_vector


    def item_forward(self, item_idx, item_features):
        """Encode items by fusing content representation and cached graph representation."""
        item_content = self._build_item_content(item_idx, item_features)
        item_graph = self._get_cached_item_graph(item_idx, item_content)
        gate = self.item_graph_gate(torch.cat([item_content, item_graph], dim=-1))
        fused = gate * item_content + (1.0 - gate) * item_graph

        item_vector = self.item_tower(fused)
        item_vector = self.output_norm(item_vector)
        return item_vector

    def forward(self, user_idx, user_features, user_color_idx, user_size_idx, item_idx, item_features):
        """Return scaled cosine-style matching scores."""
        user_vector = self.user_forward(user_idx, user_features, user_color_idx, user_size_idx)
        item_vector = self.item_forward(item_idx, item_features)

        normalized_user = F.normalize(user_vector, p=2, dim=-1)
        normalized_item = F.normalize(item_vector, p=2, dim=-1)
        return torch.sum(normalized_user * normalized_item, dim=-1, keepdim=True) / self.temperature
