import torch
import torch.optim as optim
import time
import numpy as np
import os
from models import CollaborativeFiltering, DeepFM, TwoTowerModel, OptimizedTwoTowerModel, SimplifiedTwoTowerModel, EfficientTwoTowerModel, HybridAttentionRecommendationNetwork, MultiModalLightGCN
from evaluator import Evaluator

class Trainer:
    def __init__(self, model, model_type, data, device, config=None):
        self.model = model.to(device)
        self.model_type = model_type
        self.data = data
        self.device = device
        self.config = config or {}
        self.evaluator = Evaluator(k_list=[5, 10, 20])
        
        self.num_users = data['num_users']
        self.num_items = data['num_items']
        
        self.setup_optimizer()
        self.setup_scheduler()
    
    def setup_optimizer(self):
        lr = self.config.get('lr', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if self.model_type == 'collaborative_filtering':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.model_type == 'deepfm':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.model_type == 'twotower':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.model_type == 'optimized_twotower':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.model_type == 'simplified_twotower':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.model_type == 'efficient_twotower':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.model_type == 'harn':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.model_type == 'lightgcn':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def setup_scheduler(self):
        patience = self.config.get('patience', 5)
        factor = self.config.get('factor', 0.5)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=factor, patience=patience, verbose=True
        )
    
    @staticmethod
    def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=-1)
        diff = pos_scores - neg_scores
        
        if torch.isnan(diff).any() or torch.isinf(diff).any():
            return torch.tensor(0.0, requires_grad=True, device=user_emb.device)
        
        diff = torch.clamp(diff, min=-10.0, max=10.0)
        loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, requires_grad=True, device=user_emb.device)
        
        return loss
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        batch_size = self.config.get('batch_size', 2048)
        
        if self.model_type == 'collaborative_filtering':
            return self._train_cf_epoch(batch_size)
        elif self.model_type == 'deepfm':
            return self._train_deepfm_epoch(batch_size)
        elif self.model_type == 'twotower':
            return self._train_twotower_epoch(batch_size)
        elif self.model_type == 'optimized_twotower':
            return self._train_twotower_epoch(batch_size)
        elif self.model_type == 'simplified_twotower':
            return self._train_twotower_epoch(batch_size)
        elif self.model_type == 'efficient_twotower':
            return self._train_twotower_epoch(batch_size)
        elif self.model_type == 'harn':
            return self._train_harn_epoch(batch_size)
        elif self.model_type == 'lightgcn':
            return self._train_lightgcn_epoch(batch_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _train_cf_epoch(self, batch_size):
        total_loss = 0
        interaction_matrix = self.data['interaction_matrix']
        num_users = interaction_matrix.shape[0]
        num_items = interaction_matrix.shape[1]
        
        for start_idx in range(0, num_users, batch_size):
            batch_user_indices = torch.arange(start_idx, min(start_idx + batch_size, num_users)).to(self.device)
            
            batch_users = []
            batch_pos_items = []
            batch_neg_items = []
            
            for i, user_idx in enumerate(batch_user_indices):
                user_pos_items = torch.where(torch.from_numpy(interaction_matrix[user_idx.item()]) == 1)[0]
                
                if len(user_pos_items) > 0:
                    num_pos = min(len(user_pos_items), 2)
                    pos_idx = user_pos_items[torch.randint(0, len(user_pos_items), (num_pos,)).to(self.device)]
                    neg_idx = torch.randint(0, num_items, (num_pos,)).to(self.device)
                    
                    for j in range(num_pos):
                        batch_users.append(user_idx)
                        batch_pos_items.append(pos_idx[j])
                        batch_neg_items.append(neg_idx[j])
            
            if len(batch_users) == 0:
                continue
            
            batch_users = torch.stack(batch_users)
            batch_pos_items = torch.stack(batch_pos_items)
            batch_neg_items = torch.stack(batch_neg_items)
            
            user_emb = self.model.user_embedding(batch_users)
            pos_item_emb = self.model.item_embedding(batch_pos_items)
            neg_item_emb = self.model.item_embedding(batch_neg_items)
            
            loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)
            
            if torch.isnan(loss):
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / (num_users // batch_size + 1)
    
    def _train_deepfm_epoch(self, batch_size):
        total_loss = 0
        interaction_matrix = self.data['interaction_matrix']
        num_users = interaction_matrix.shape[0]
        num_items = interaction_matrix.shape[1]
        
        for start_idx in range(0, num_users, batch_size):
            batch_user_indices = torch.arange(start_idx, min(start_idx + batch_size, num_users)).to(self.device)
            
            batch_users = []
            batch_pos_items = []
            batch_neg_items = []
            
            for i, user_idx in enumerate(batch_user_indices):
                user_pos_items = torch.where(torch.from_numpy(interaction_matrix[user_idx.item()]) == 1)[0]
                
                if len(user_pos_items) > 0:
                    num_pos = min(len(user_pos_items), 2)
                    pos_idx = user_pos_items[torch.randint(0, len(user_pos_items), (num_pos,)).to(self.device)]
                    neg_idx = torch.randint(0, num_items, (num_pos,)).to(self.device)
                    
                    for j in range(num_pos):
                        batch_users.append(user_idx)
                        batch_pos_items.append(pos_idx[j])
                        batch_neg_items.append(neg_idx[j])
            
            if len(batch_users) == 0:
                continue
            
            batch_users = torch.stack(batch_users)
            batch_pos_items = torch.stack(batch_pos_items)
            batch_neg_items = torch.stack(batch_neg_items)
            
            user_feat = self.data['user_features_tensor'][batch_users]
            user_color = self.data['user_color_indices_tensor'][batch_users]
            user_size = self.data['user_size_indices_tensor'][batch_users]
            pos_item_feat = self.data['item_features_tensor'][batch_pos_items]
            neg_item_feat = self.data['item_features_tensor'][batch_neg_items]
            
            pos_scores = self.model(batch_users, user_feat, user_color, user_size, batch_pos_items, pos_item_feat)
            neg_scores = self.model(batch_users, user_feat, user_color, user_size, batch_neg_items, neg_item_feat)
            
            user_emb = self.model.user_embedding(batch_users)
            pos_item_emb = self.model.item_embedding(batch_pos_items)
            neg_item_emb = self.model.item_embedding(batch_neg_items)
            
            loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)
            
            if torch.isnan(loss):
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / (num_users // batch_size + 1)
    
    def _train_twotower_epoch(self, batch_size):
        total_loss = 0
        interaction_matrix = self.data['interaction_matrix']
        num_users = interaction_matrix.shape[0]
        num_items = interaction_matrix.shape[1]
        
        for start_idx in range(0, num_users, batch_size):
            batch_user_indices = torch.arange(start_idx, min(start_idx + batch_size, num_users)).to(self.device)
            
            batch_users = []
            batch_pos_items = []
            batch_neg_items = []
            
            for i, user_idx in enumerate(batch_user_indices):
                user_pos_items = torch.where(torch.from_numpy(interaction_matrix[user_idx.item()]) == 1)[0]
                
                if len(user_pos_items) > 0:
                    num_pos = min(len(user_pos_items), 2)
                    pos_idx = user_pos_items[torch.randint(0, len(user_pos_items), (num_pos,)).to(self.device)]
                    neg_idx = torch.randint(0, num_items, (num_pos,)).to(self.device)
                    
                    for j in range(num_pos):
                        batch_users.append(user_idx)
                        batch_pos_items.append(pos_idx[j])
                        batch_neg_items.append(neg_idx[j])
            
            if len(batch_users) == 0:
                continue
            
            batch_users = torch.stack(batch_users)
            batch_pos_items = torch.stack(batch_pos_items)
            batch_neg_items = torch.stack(batch_neg_items)
            
            user_feat = self.data['user_features_tensor'][batch_users]
            user_color = self.data['user_color_indices_tensor'][batch_users]
            user_size = self.data['user_size_indices_tensor'][batch_users]
            pos_item_feat = self.data['item_features_tensor'][batch_pos_items]
            neg_item_feat = self.data['item_features_tensor'][batch_neg_items]
            
            user_emb = self.model.user_forward(batch_users, user_feat, user_color, user_size)
            pos_item_emb = self.model.item_forward(batch_pos_items, pos_item_feat)
            neg_item_emb = self.model.item_forward(batch_neg_items, neg_item_feat)
            
            loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)
            
            if torch.isnan(loss):
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / (num_users // batch_size + 1)
    
    def _train_lightgcn_epoch(self, batch_size):
        total_loss = 0
        edge_index = self.data['user_item_edge_index']
        num_edges = edge_index.shape[1]
        num_items = self.data['num_items']
        
        indices = torch.randperm(num_edges)
        
        for start_idx in range(0, num_edges, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            user_indices = edge_index[0, batch_indices]
            pos_item_indices = edge_index[1, batch_indices] - self.num_users
            
            neg_item_indices = torch.randint(0, num_items, (len(user_indices),)).to(self.device)
            
            user_emb, item_emb = self.model(self.data['edge_index'], 
                                          self.data['user_features_tensor'],
                                          self.data['user_color_indices_tensor'],
                                          self.data['user_size_indices_tensor'],
                                          self.data['item_features_tensor'])
            
            user_batch_emb = user_emb[user_indices]
            pos_item_batch_emb = item_emb[pos_item_indices]
            neg_item_batch_emb = item_emb[neg_item_indices]
            
            loss = self.bpr_loss(user_batch_emb, pos_item_batch_emb, neg_item_batch_emb)
            
            if torch.isnan(loss):
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / (num_edges // batch_size + 1) if num_edges > 0 else 0.0
    
    def _train_harn_epoch(self, batch_size):
        total_loss = 0
        interaction_matrix = self.data['interaction_matrix']
        num_users = interaction_matrix.shape[0]
        num_items = interaction_matrix.shape[1]
        
        for start_idx in range(0, num_users, batch_size):
            batch_user_indices = torch.arange(start_idx, min(start_idx + batch_size, num_users)).to(self.device)
            
            batch_users = []
            batch_pos_items = []
            batch_neg_items = []
            
            for i, user_idx in enumerate(batch_user_indices):
                user_pos_items = torch.where(torch.from_numpy(interaction_matrix[user_idx.item()]) == 1)[0]
                
                if len(user_pos_items) > 0:
                    num_pos = min(len(user_pos_items), 2)
                    pos_idx = user_pos_items[torch.randint(0, len(user_pos_items), (num_pos,)).to(self.device)]
                    neg_idx = torch.randint(0, num_items, (num_pos,)).to(self.device)
                    
                    for j in range(num_pos):
                        batch_users.append(user_idx)
                        batch_pos_items.append(pos_idx[j])
                        batch_neg_items.append(neg_idx[j])
            
            if len(batch_users) == 0:
                continue
            
            batch_users = torch.stack(batch_users)
            batch_pos_items = torch.stack(batch_pos_items)
            batch_neg_items = torch.stack(batch_neg_items)
            
            user_feat = self.data['user_features_tensor'][batch_users]
            user_color = self.data['user_color_indices_tensor'][batch_users]
            user_size = self.data['user_size_indices_tensor'][batch_users]
            pos_item_feat = self.data['item_features_tensor'][batch_pos_items]
            neg_item_feat = self.data['item_features_tensor'][batch_neg_items]
            
            interaction_vector = self.model.get_user_item_interaction(
                batch_users, user_feat, user_color, user_size, 
                batch_pos_items, pos_item_feat
            )
            
            pos_item_interaction = self.model.get_user_item_interaction(
                batch_users, user_feat, user_color, user_size, 
                batch_pos_items, pos_item_feat
            )
            
            neg_item_interaction = self.model.get_user_item_interaction(
                batch_users, user_feat, user_color, user_size, 
                batch_neg_items, neg_item_feat
            )
            
            loss = self.bpr_loss(pos_item_interaction, pos_item_interaction, neg_item_interaction)
            
            if torch.isnan(loss):
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / (num_users // batch_size + 1)
    
    def train(self, epochs=100, eval_every=10, early_stopping_patience=10):
        print(f"\nStarting training {self.model_type}...")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}, Eval every: {eval_every}")
        
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        best_ndcg = 0
        patience_counter = 0
        training_times = []
        
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            train_loss = self.train_epoch()
            
            epoch_time = time.time() - epoch_start_time
            training_times.append(epoch_time)
            
            if epoch % eval_every == 0:
                eval_results, inference_time = self.evaluator.evaluate_model(
                    self.model, self.model_type, self.data, self.device
                )
                
                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Epoch Time: {epoch_time:.2f}s")
                print(f"  Inference Time: {inference_time:.2f}s")
                
                for k in [5, 10, 20]:
                    print(f"  Top-{k}: Precision={eval_results[k]['precision']:.4f}, "
                          f"Recall={eval_results[k]['recall']:.4f}, NDCG={eval_results[k]['ndcg']:.4f}")
                
                current_ndcg = eval_results[10]['ndcg']
                self.scheduler.step(train_loss)
                
                if current_ndcg > best_ndcg:
                    best_ndcg = current_ndcg
                    patience_counter = 0
                    torch.save(self.model.state_dict(), f'checkpoints/best_{self.model_type}_model.pth')
                    print(f"  ✅ New best model saved! NDCG@10: {best_ndcg:.4f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        total_training_time = time.time() - total_start_time
        avg_epoch_time = np.mean(training_times)
        
        print(f"\n=== Training Summary ===")
        print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f}min)")
        print(f"Average epoch time: {avg_epoch_time:.2f}s")
        print(f"Total epochs: {len(training_times)}")
        
        return {
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time,
            'total_epochs': len(training_times),
            'best_ndcg': best_ndcg
        }
    
    def final_evaluation(self):
        print(f"\n=== Final Evaluation ===")
        
        checkpoint_path = f'checkpoints/best_{self.model_type}_model.pth'
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded best model from {checkpoint_path}")
        
        final_results, inference_time = self.evaluator.evaluate_model(
            self.model, self.model_type, self.data, self.device
        )
        
        print(f"Final inference time: {inference_time:.2f}s")
        print(f"Total test items: {self.data['total_test_items']}")
        
        print("\nFinal Results:")
        for k in [5, 10, 20]:
            print(f"Top-{k}:")
            print(f"  Precision@{k}: {final_results[k]['precision']:.4f}")
            print(f"  Recall@{k}: {final_results[k]['recall']:.4f}")
            print(f"  NDCG@{k}: {final_results[k]['ndcg']:.4f}")
        
        results_df = self.evaluator.save_results(final_results, self.model_type, inference_time)
        
        return {
            'results': final_results,
            'inference_time': inference_time,
            'results_df': results_df
        }