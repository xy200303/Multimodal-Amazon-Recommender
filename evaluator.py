"""Top-K 推荐实验评测工具。

该模块负责：
1. 在全量商品上为每个测试用户打分；
2. 屏蔽训练阶段已见商品；
3. 计算 Precision@K、Recall@K、NDCG@K 等指标。
"""

import torch
import numpy as np
import time
import pandas as pd
import torch.nn.functional as F

class Evaluator:
    def __init__(self, k_list=[5, 10, 20]):
        """初始化评测器，并指定需要统计的 K 值。"""
        self.k_list = k_list

    @staticmethod
    def mask_seen_items(scores, interaction_matrix, user_idx):
        """屏蔽用户训练阶段已交互商品，避免重复推荐。"""
        if interaction_matrix is None:
            return scores

        masked_scores = scores.clone()
        seen_mask = torch.from_numpy(interaction_matrix[user_idx] > 0).to(masked_scores.device)
        masked_scores[seen_mask] = float('-inf')
        return masked_scores
    
    @staticmethod
    def calculate_metrics(top_k_items, test_item_indices, k):
        """计算单个用户在指定 K 下的排序指标。"""
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
        """根据模型类型选择评测策略，并聚合所有测试用户结果。"""
        if model_type in ['random_baseline', 'popularity_baseline']:
            return self._evaluate_baseline(model, model_type, test_data, device)
        
        model.eval()
        
        results = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in self.k_list}
        
        test_users_list = test_data['test_users_list']
        test_items_list = test_data['test_items_list']
        num_users = test_data['num_users']
        num_items = test_data['num_items']
        interaction_matrix = test_data.get('interaction_matrix')
        
        inference_start_time = time.time()
        
        # 评测涉及全量商品，因此优先使用批量打分策略。
        if model_type in ['twotower', 'graph_augmented_twotower']:
            all_scores = self._evaluate_twotower_batch(model, test_data, num_items, device)
        elif model_type == 'lightgcn':
            all_scores = self._evaluate_lightgcn_batch(model, test_data, num_items, device)
        elif model_type == 'collaborative_filtering':
            all_scores = self._evaluate_cf_batch(model, test_users_list, num_items, device)
        elif model_type in ['deepfm', 'harn']:
            all_scores = self._evaluate_interaction_batch(model, model_type, test_data, num_items, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if isinstance(all_scores, torch.Tensor) and interaction_matrix is not None:
            seen_mask = torch.from_numpy(
                interaction_matrix[np.array(test_users_list)] > 0
            ).to(all_scores.device)
            all_scores = all_scores.masked_fill(seen_mask, float('-inf'))

        for i, (user_idx, test_item_indices) in enumerate(zip(test_users_list, test_items_list)):
            if len(test_item_indices) == 0:
                continue

            scores = all_scores[i]
            if not isinstance(all_scores, torch.Tensor):
                scores = self.mask_seen_items(scores, interaction_matrix, user_idx)
            
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
    
    def _evaluate_twotower_batch(self, model, test_data, num_items, device):
        """Score all test users against all items using dot products in tower space."""
        with torch.inference_mode():
            if hasattr(model, 'refresh_graph_cache'):
                model.refresh_graph_cache(
                    test_data['edge_index'],
                    test_data['user_features_tensor'],
                    test_data['user_color_indices_tensor'],
                    test_data['user_size_indices_tensor'],
                    test_data['item_features_tensor']
                )

            all_item_indices = torch.arange(num_items, device=device)
            item_vectors = model.item_forward(all_item_indices, test_data['item_features_tensor'])
            item_vectors = F.normalize(item_vectors, p=2, dim=-1)

            test_users = torch.tensor(test_data['test_users_list'], device=device, dtype=torch.long)
            user_vectors = model.user_forward(
                test_users,
                test_data['user_features_tensor'][test_users],
                test_data['user_color_indices_tensor'][test_users],
                test_data['user_size_indices_tensor'][test_users]
            )
            user_vectors = F.normalize(user_vectors, p=2, dim=-1)

            scores = torch.matmul(user_vectors, item_vectors.t())
            if hasattr(model, 'temperature'):
                temperature = getattr(model, 'temperature')
                if torch.is_tensor(temperature):
                    temperature = torch.clamp(temperature.to(device).abs(), min=1e-6)
                else:
                    temperature = torch.tensor(float(temperature), device=device)
                scores = scores / temperature

            if hasattr(model, 'clear_graph_cache'):
                model.clear_graph_cache()

            return scores
    
    def _evaluate_lightgcn_batch(self, model, test_data, num_items, device):
        """Score all test users with the graph-propagated embeddings."""
        with torch.inference_mode():
            user_emb, item_emb = model(test_data['edge_index'], 
                                      test_data['user_features_tensor'],
                                      test_data['user_color_indices_tensor'], 
                                      test_data['user_size_indices_tensor'],
                                      test_data['item_features_tensor'])

            test_users = torch.tensor(test_data['test_users_list'], device=device, dtype=torch.long)
            return torch.matmul(user_emb[test_users], item_emb.t())
    
    def _evaluate_cf_batch(self, model, test_users_list, num_items, device):
        """Score all items for each test user with matrix-factorization embeddings."""
        with torch.inference_mode():
            test_users = torch.tensor(test_users_list, device=device, dtype=torch.long)
            all_item_indices = torch.arange(num_items, device=device)
            user_vectors = model.user_embedding(test_users)
            item_vectors = model.item_embedding(all_item_indices)
            return torch.matmul(user_vectors, item_vectors.t())
    
    def _evaluate_interaction_batch(self, model, model_type, test_data, num_items, device):
        """Batch score interaction-based models that need explicit user-item pairs."""
        with torch.inference_mode():
            all_item_indices = torch.arange(num_items, device=device)
            item_feat = test_data['item_features_tensor']
            test_users = torch.tensor(test_data['test_users_list'], device=device, dtype=torch.long)

            user_batch_size = 32
            item_batch_size = 512
            test_users_list = test_data['test_users_list']
            all_scores = []

            for user_start in range(0, len(test_users_list), user_batch_size):
                user_batch = test_users[user_start:user_start + user_batch_size]
                current_user_batch_size = user_batch.size(0)

                user_feat = test_data['user_features_tensor'][user_batch]
                user_color = test_data['user_color_indices_tensor'][user_batch]
                user_size = test_data['user_size_indices_tensor'][user_batch]

                batch_score_parts = []
                for item_start in range(0, num_items, item_batch_size):
                    item_batch = all_item_indices[item_start:item_start + item_batch_size]
                    item_feat_batch = item_feat[item_batch]
                    current_item_batch_size = item_batch.size(0)

                    expanded_user_idx = user_batch.unsqueeze(1).expand(
                        current_user_batch_size, current_item_batch_size
                    ).reshape(-1)
                    expanded_user_feat = user_feat.unsqueeze(1).expand(
                        current_user_batch_size, current_item_batch_size, -1
                    ).reshape(-1, user_feat.size(-1))
                    expanded_user_color = user_color.unsqueeze(1).expand(
                        current_user_batch_size, current_item_batch_size
                    ).reshape(-1)
                    expanded_user_size = user_size.unsqueeze(1).expand(
                        current_user_batch_size, current_item_batch_size
                    ).reshape(-1)
                    expanded_item_idx = item_batch.unsqueeze(0).expand(
                        current_user_batch_size, current_item_batch_size
                    ).reshape(-1)
                    expanded_item_feat = item_feat_batch.unsqueeze(0).expand(
                        current_user_batch_size, current_item_batch_size, -1
                    ).reshape(-1, item_feat_batch.size(-1))

                    scores = model(
                        expanded_user_idx,
                        expanded_user_feat,
                        expanded_user_color,
                        expanded_user_size,
                        expanded_item_idx,
                        expanded_item_feat
                    ).view(current_user_batch_size, current_item_batch_size)
                    batch_score_parts.append(scores)

                all_scores.append(torch.cat(batch_score_parts, dim=1))

            return torch.cat(all_scores, dim=0)
    
    def _evaluate_baseline(self, model, model_type, test_data, device):
        """Evaluate simple non-neural baselines under the same full-ranking protocol."""
        results = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in self.k_list}
        
        test_users_list = test_data['test_users_list']
        test_items_list = test_data['test_items_list']
        num_items = test_data['num_items']
        interaction_matrix = test_data.get('interaction_matrix')
        
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

            scores = self.mask_seen_items(scores, interaction_matrix, user_idx)
            
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
    
    def save_results(self, results, model_name, inference_time, output_dir='results'):
        """Persist evaluation metrics to CSV for later reporting and thesis tables."""
        results_df = pd.DataFrame({
            'K': self.k_list,
            'Precision': [results[k]['precision'] for k in self.k_list],
            'Recall': [results[k]['recall'] for k in self.k_list],
            'NDCG': [results[k]['ndcg'] for k in self.k_list]
        })
        
        results_df.to_csv(f'{output_dir}/{model_name}_evaluation_results.csv', index=False)
        print(f"Results saved to {output_dir}/{model_name}_evaluation_results.csv")
        
        return results_df
