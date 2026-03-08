import torch
import numpy as np

class PopularityBaseline:
    def __init__(self, num_items):
        self.num_items = num_items
        self.name = "Popularity Baseline"
        self.popularity_scores = None
    
    def fit(self, interaction_matrix):
        self.popularity_scores = torch.FloatTensor(interaction_matrix.sum(axis=0))
    
    def predict(self, num_samples=1):
        if self.popularity_scores is None:
            raise ValueError("Model must be fitted before prediction")
        _, top_indices = torch.topk(self.popularity_scores, num_samples)
        return top_indices
    
    def get_scores(self, num_items):
        if self.popularity_scores is None:
            raise ValueError("Model must be fitted before prediction")
        return self.popularity_scores[:num_items]