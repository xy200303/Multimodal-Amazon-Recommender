"""流行度推荐基线。"""

import torch

class PopularityBaseline:
    """根据训练集中商品出现次数进行排序的基线模型。"""
    def __init__(self, num_items):
        self.num_items = num_items
        self.name = "Popularity Baseline"
        self.popularity_scores = None
    
    def fit(self, interaction_matrix):
        """统计每个商品在训练集中被交互的次数。"""
        self.popularity_scores = torch.FloatTensor(interaction_matrix.sum(axis=0))
    
    def predict(self, num_samples=1):
        """返回最热门的若干商品索引。"""
        if self.popularity_scores is None:
            raise ValueError("Model must be fitted before prediction")
        _, top_indices = torch.topk(self.popularity_scores, num_samples)
        return top_indices
    
    def get_scores(self, num_items):
        """返回全量商品的流行度分数。"""
        if self.popularity_scores is None:
            raise ValueError("Model must be fitted before prediction")
        return self.popularity_scores[:num_items]
