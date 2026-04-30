"""随机推荐基线。"""

import torch

class RandomBaseline:
    """为每次推荐请求随机生成商品分数。"""
    def __init__(self, num_items):
        self.num_items = num_items
        self.name = "Random Baseline"
    
    def predict(self, num_samples=1):
        """随机采样若干商品索引。"""
        return torch.randint(0, self.num_items, (num_samples,))
    
    def fit(self, *args, **kwargs):
        """随机基线无需训练，保留空接口以统一调用方式。"""
        pass
    
    def get_scores(self, num_items):
        """返回随机分数向量，用于 Top-K 排序。"""
        return torch.rand(num_items)
