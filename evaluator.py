import torch
import numpy as np
import time
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, k_list=[5, 10, 20]):
        self.k_list = k_list
    
    @staticmethod
    def calculate_metrics(top_k_items, test_item_indices, k):
        hit_count = sum(1 for item_idx in top_k_items if item_idx in test_item_indices)
        
        precision = hit_count / k
        recall = hit_count / len(test_item_indices)
        
        dcg = 0.0
        for i, item_idx in enumerate(top_k_items):
            if item_idx in test_item_indices:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_item_indices), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'ndcg': ndcg,
            'hit_count': hit_count
        }
    
    def evaluate_model(self, model, model_type, test_data, device):
        if model_type in ['random_baseline', 'popularity_baseline']:
            return self._evaluate_baseline(model, model_type, test_data, device)
        
        model.eval()
        
        results = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in self.k_list}
        
        test_users_list = test_data['test_users_list']
        test_items_list = test_data['test_items_list']
        num_users = test_data['num_users']
        num_items = test_data['num_items']
        
        inference_start_time = time.time()
        
        for user_idx, test_item_indices in zip(test_users_list, test_items_list):
            if len(test_item_indices) == 0:
                continue
            
            if model_type == 'collaborative_filtering':
                scores = self._evaluate_cf(model, user_idx, num_items, device)
            elif model_type == 'deepfm':
                scores = self._evaluate_deepfm(model, user_idx, test_data, num_items, device)
            elif model_type == 'twotower':
                scores = self._evaluate_twotower(model, user_idx, test_data, num_items, device)
            elif model_type == 'optimized_twotower':
                scores = self._evaluate_twotower(model, user_idx, test_data, num_items, device)
            elif model_type == 'simplified_twotower':
                scores = self._evaluate_twotower(model, user_idx, test_data, num_items, device)
            elif model_type == 'efficient_twotower':
                scores = self._evaluate_twotower(model, user_idx, test_data, num_items, device)
            elif model_type == 'harn':
                scores = self._evaluate_harn(model, user_idx, test_data, num_items, device)
            elif model_type == 'lightgcn':
                scores = self._evaluate_lightgcn(model, user_idx, test_data, num_items, device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            for k in self.k_list:
                _, top_k_indices = torch.topk(scores, k)
                top_k_items = top_k_indices.cpu().numpy().tolist()
                
                metrics = self.calculate_metrics(top_k_items, test_item_indices, k)
                results[k]['precision'].append(metrics['precision'])
                results[k]['recall'].append(metrics['recall'])
                results[k]['ndcg'].append(metrics['ndcg'])
        
        inference_time = time.time() - inference_start_time
        
        avg_results = {}
        for k in self.k_list:
            avg_results[k] = {
                'precision': np.mean(results[k]['precision']) if results[k]['precision'] else 0,
                'recall': np.mean(results[k]['recall']) if results[k]['recall'] else 0,
                'ndcg': np.mean(results[k]['ndcg']) if results[k]['ndcg'] else 0
            }
        
        return avg_results, inference_time
    
    def _evaluate_baseline(self, model, model_type, test_data, device):
        results = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in self.k_list}
        
        test_users_list = test_data['test_users_list']
        test_items_list = test_data['test_items_list']
        num_items = test_data['num_items']
        
        inference_start_time = time.time()
        
        if model_type == 'popularity_baseline':
            scores = model.get_scores(num_items)
        
        for user_idx, test_item_indices in zip(test_users_list, test_items_list):
            if len(test_item_indices) == 0:
                continue
            
            if model_type == 'random_baseline':
                scores = model.get_scores(num_items)
            elif model_type == 'popularity_baseline':
                scores = model.get_scores(num_items)
            
            for k in self.k_list:
                _, top_k_indices = torch.topk(scores, k)
                top_k_items = top_k_indices.cpu().numpy().tolist()
                
                metrics = self.calculate_metrics(top_k_items, test_item_indices, k)
                results[k]['precision'].append(metrics['precision'])
                results[k]['recall'].append(metrics['recall'])
                results[k]['ndcg'].append(metrics['ndcg'])
        
        inference_time = time.time() - inference_start_time
        
        avg_results = {}
        for k in self.k_list:
            avg_results[k] = {
                'precision': np.mean(results[k]['precision']) if results[k]['precision'] else 0,
                'recall': np.mean(results[k]['recall']) if results[k]['recall'] else 0,
                'ndcg': np.mean(results[k]['ndcg']) if results[k]['ndcg'] else 0
            }
        
        return avg_results, inference_time
    
    def _evaluate_cf(self, model, user_idx, num_items, device):
        all_item_indices = torch.arange(num_items, device=device)
        user_idx_tensor = torch.tensor([user_idx], device=device).expand(num_items)
        scores = model(user_idx_tensor, all_item_indices)
        return scores.squeeze()
    
    def _evaluate_deepfm(self, model, user_idx, test_data, num_items, device):
        user_feat = test_data['user_features_tensor'][user_idx].unsqueeze(0)
        user_color = test_data['user_color_indices_tensor'][user_idx].unsqueeze(0)
        user_size = test_data['user_size_indices_tensor'][user_idx].unsqueeze(0)
        
        all_item_indices = torch.arange(num_items, device=device)
        item_feat = test_data['item_features_tensor'].to(device)
        
        user_idx_tensor = torch.tensor([user_idx], device=device).expand(num_items)
        user_feat_batch = user_feat.expand(num_items, -1)
        user_color_batch = user_color.expand(num_items)
        user_size_batch = user_size.expand(num_items)
        
        scores = model(user_idx_tensor, user_feat_batch, user_color_batch, user_size_batch, 
                      all_item_indices, item_feat)
        return scores.squeeze()
    
    def _evaluate_twotower(self, model, user_idx, test_data, num_items, device):
        user_feat = test_data['user_features_tensor'][user_idx].unsqueeze(0)
        user_color = test_data['user_color_indices_tensor'][user_idx].unsqueeze(0)
        user_size = test_data['user_size_indices_tensor'][user_idx].unsqueeze(0)
        
        user_vector = model.user_forward(torch.tensor([user_idx], device=device), 
                                      user_feat, user_color, user_size)
        
        all_item_indices = torch.arange(num_items, device=device)
        item_vectors = model.item_forward(all_item_indices, test_data['item_features_tensor'])
        
        scores = (user_vector * item_vectors).sum(dim=-1)
        return scores
    
    def _evaluate_harn(self, model, user_idx, test_data, num_items, device):
        user_feat = test_data['user_features_tensor'][user_idx].unsqueeze(0)
        user_color = test_data['user_color_indices_tensor'][user_idx].unsqueeze(0)
        user_size = test_data['user_size_indices_tensor'][user_idx].unsqueeze(0)
        
        all_item_indices = torch.arange(num_items, device=device)
        item_feat = test_data['item_features_tensor'].to(device)
        
        user_idx_tensor = torch.tensor([user_idx], device=device).expand(num_items)
        user_feat_batch = user_feat.expand(num_items, -1)
        user_color_batch = user_color.expand(num_items)
        user_size_batch = user_size.expand(num_items)
        
        scores = model(user_idx_tensor, user_feat_batch, user_color_batch, user_size_batch, 
                      all_item_indices, item_feat)
        return scores.squeeze()
    
    def _evaluate_lightgcn(self, model, user_idx, test_data, num_items, device):
        user_emb, item_emb = model(test_data['edge_index'], test_data['user_features_tensor'],
                                  test_data['user_color_indices_tensor'], 
                                  test_data['user_size_indices_tensor'],
                                  test_data['item_features_tensor'])
        
        user_vector = user_emb[user_idx]
        scores = (user_vector.unsqueeze(0) * item_emb).sum(dim=-1)
        return scores
    
    def save_results(self, results, model_name, inference_time, output_dir='results'):
        results_df = pd.DataFrame({
            'K': self.k_list,
            'Precision': [results[k]['precision'] for k in self.k_list],
            'Recall': [results[k]['recall'] for k in self.k_list],
            'NDCG': [results[k]['ndcg'] for k in self.k_list]
        })
        
        results_df.to_csv(f'{output_dir}/{model_name}_evaluation_results.csv', index=False)
        print(f"Results saved to {output_dir}/{model_name}_evaluation_results.csv")
        
        return results_df