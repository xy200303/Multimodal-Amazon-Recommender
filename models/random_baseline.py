import torch
import numpy as np

class RandomBaseline:
    def __init__(self, num_items):
        self.num_items = num_items
        self.name = "Random Baseline"
    
    def predict(self, num_samples=1):
        return torch.randint(0, self.num_items, (num_samples,))
    
    def fit(self, *args, **kwargs):
        pass
    
    def get_scores(self, num_items):
        return torch.rand(num_items)