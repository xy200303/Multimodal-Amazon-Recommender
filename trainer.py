"""模型训练器。

该模块为项目中的全部可训练推荐模型提供统一训练接口，主要负责：
1. 优化器与学习率调度器配置；
2. 负采样；
3. 单轮训练分发；
4. 训练历史与 checkpoint 保存；
5. 训练结束后的最终评测。
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from evaluator import Evaluator

TWO_TOWER_MODEL_TYPES = {
    'twotower'
}

GRAPH_AUGMENTED_MODEL_TYPES = {
    'graph_augmented_twotower'
}

SUPPORTED_MODEL_TYPES = {
    'collaborative_filtering',
    'deepfm',
    'twotower',
    'graph_augmented_twotower',
    'harn',
    'lightgcn'
}


class Trainer:
    """统一封装模型训练、保存与最终评测流程。"""

    def __init__(self, model, model_type, data, device, config=None):
        """缓存训练过程中会重复使用的张量和配置。"""
        self.model = model.to(device)
        self.model_type = model_type
        self.run_name = (config or {}).get('run_name', model_type)
        self.data = data
        self.device = device
        self.config = config or {}
        self.evaluator = Evaluator(k_list=[5, 10, 20])

        self.num_users = data['num_users']
        self.num_items = data['num_items']

        # interaction_mask 用于负采样时快速过滤用户已交互商品。
        self.interaction_mask = torch.from_numpy(data['interaction_matrix'] > 0).to(device)

        # user_item_edge_index 中只包含训练集正样本交互。
        self.train_user_indices = data['user_item_edge_index'][0].to(device)
        self.train_item_indices = (data['user_item_edge_index'][1] - self.num_users).to(device)

        # LightGCN 训练和评测都要用到整张图。
        self.edge_index = data['edge_index'].to(device)

        self.setup_optimizer()
        self.setup_scheduler()

    def setup_optimizer(self):
        """为所有支持训练的模型统一配置 Adam 优化器。"""
        lr = self.config.get('lr', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-5)

        if self.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def setup_scheduler(self):
        """当训练损失进入平台期时自动降低学习率。"""
        patience = self.config.get('patience', 5)
        factor = self.config.get('factor', 0.5)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            verbose=True
        )

    @staticmethod
    def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
        """BPR 排序损失，适用于隐式反馈场景。"""
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

    @staticmethod
    def bce_loss(pos_scores, neg_scores):
        """对正负样本对执行二分类优化时使用的损失。"""
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pos_scores.squeeze(), pos_labels.squeeze()
        )
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            neg_scores.squeeze(), neg_labels.squeeze()
        )

        loss = pos_loss + neg_loss

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, requires_grad=True, device=pos_scores.device)

        return loss

    @staticmethod
    def contrastive_loss(user_emb, pos_item_emb, neg_item_emb, temperature=0.1):
        """双塔召回模型常用的对比学习损失。"""
        if torch.is_tensor(temperature):
            temperature = temperature.to(user_emb.device)
            temperature = torch.clamp(temperature.abs(), min=1e-6)
        else:
            temperature = torch.tensor(float(temperature), device=user_emb.device)

        pos_sim = torch.cosine_similarity(user_emb, pos_item_emb, dim=-1)
        neg_sim = torch.cosine_similarity(user_emb, neg_item_emb, dim=-1)

        pos_scores = pos_sim / temperature
        neg_scores = neg_sim / temperature

        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-8)

        loss = pos_loss.mean() + neg_loss.mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, requires_grad=True, device=user_emb.device)

        return loss

    def iterate_training_batches(self, batch_size):
        """对打乱后的训练正样本交互按批次迭代。"""
        num_samples = self.train_user_indices.size(0)
        permutation = torch.randperm(num_samples, device=self.device)

        for start_idx in range(0, num_samples, batch_size):
            batch_indices = permutation[start_idx:start_idx + batch_size]
            yield self.train_user_indices[batch_indices], self.train_item_indices[batch_indices]

    def sample_negative_items_for_users(self, user_indices):
        """为每个正样本交互采一个负样本，并避开用户已交互商品。"""
        neg_item_indices = torch.randint(0, self.num_items, (user_indices.size(0),), device=self.device)
        invalid_mask = self.interaction_mask[user_indices, neg_item_indices]

        while invalid_mask.any():
            neg_item_indices[invalid_mask] = torch.randint(
                0, self.num_items, (int(invalid_mask.sum().item()),), device=self.device
            )
            invalid_mask = self.interaction_mask[user_indices, neg_item_indices]

        return neg_item_indices

    def train_epoch(self):
        """根据模型类型选择对应的单轮训练逻辑。"""
        self.model.train()
        batch_size = self.config.get('batch_size', 2048)

        if self.model_type == 'collaborative_filtering':
            return self._train_cf_epoch(batch_size)
        if self.model_type == 'deepfm':
            return self._train_deepfm_epoch(batch_size)
        if self.model_type in GRAPH_AUGMENTED_MODEL_TYPES:
            return self._train_graph_augmented_twotower_epoch(batch_size)
        if self.model_type in TWO_TOWER_MODEL_TYPES:
            return self._train_twotower_epoch(batch_size)
        if self.model_type == 'harn':
            return self._train_harn_epoch(batch_size)
        if self.model_type == 'lightgcn':
            return self._train_lightgcn_epoch(batch_size)

        raise ValueError(f"Unknown model type: {self.model_type}")

    def _train_cf_epoch(self, batch_size):
        """协同过滤模型的批量训练逻辑。"""
        total_loss = 0.0
        total_samples = 0

        for batch_users, batch_pos_items in self.iterate_training_batches(batch_size):
            batch_neg_items = self.sample_negative_items_for_users(batch_users)

            user_emb = self.model.user_embedding(batch_users)
            pos_item_emb = self.model.item_embedding(batch_pos_items)
            neg_item_emb = self.model.item_embedding(batch_neg_items)

            loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)
            if torch.isnan(loss):
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            current_batch_size = batch_users.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        return total_loss / max(total_samples, 1)

    def _train_deepfm_epoch(self, batch_size):
        """DeepFM 模型的正负样本配对训练逻辑。"""
        total_loss = 0.0
        total_samples = 0

        for batch_users, batch_pos_items in self.iterate_training_batches(batch_size):
            batch_neg_items = self.sample_negative_items_for_users(batch_users)

            user_feat = self.data['user_features_tensor'][batch_users]
            user_color = self.data['user_color_indices_tensor'][batch_users]
            user_size = self.data['user_size_indices_tensor'][batch_users]
            pos_item_feat = self.data['item_features_tensor'][batch_pos_items]
            neg_item_feat = self.data['item_features_tensor'][batch_neg_items]

            pos_scores = self.model(
                batch_users, user_feat, user_color, user_size, batch_pos_items, pos_item_feat
            )
            neg_scores = self.model(
                batch_users, user_feat, user_color, user_size, batch_neg_items, neg_item_feat
            )

            loss = self.bce_loss(pos_scores, neg_scores)
            if torch.isnan(loss):
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            current_batch_size = batch_users.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        return total_loss / max(total_samples, 1)

    def _train_twotower_epoch(self, batch_size):
        """双塔系列模型基于正负样本对的对比训练逻辑。"""
        total_loss = 0.0
        total_samples = 0

        for batch_users, batch_pos_items in self.iterate_training_batches(batch_size):
            batch_neg_items = self.sample_negative_items_for_users(batch_users)

            user_feat = self.data['user_features_tensor'][batch_users]
            user_color = self.data['user_color_indices_tensor'][batch_users]
            user_size = self.data['user_size_indices_tensor'][batch_users]
            pos_item_feat = self.data['item_features_tensor'][batch_pos_items]
            neg_item_feat = self.data['item_features_tensor'][batch_neg_items]

            user_emb = self.model.user_forward(batch_users, user_feat, user_color, user_size)
            pos_item_emb = self.model.item_forward(batch_pos_items, pos_item_feat)
            neg_item_emb = self.model.item_forward(batch_neg_items, neg_item_feat)

            temperature = getattr(self.model, 'temperature', 0.1)
            loss = self.contrastive_loss(user_emb, pos_item_emb, neg_item_emb, temperature=temperature)
            if torch.isnan(loss):
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            current_batch_size = batch_users.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        return total_loss / max(total_samples, 1)

    def _train_graph_augmented_twotower_epoch(self, batch_size):
        """图增强双塔模型在每个 epoch 开始前刷新一次全图缓存。"""
        total_loss = 0.0
        total_samples = 0

        self.model.refresh_graph_cache(
            self.edge_index,
            self.data['user_features_tensor'],
            self.data['user_color_indices_tensor'],
            self.data['user_size_indices_tensor'],
            self.data['item_features_tensor']
        )

        for batch_users, batch_pos_items in self.iterate_training_batches(batch_size):
            batch_neg_items = self.sample_negative_items_for_users(batch_users)

            user_feat = self.data['user_features_tensor'][batch_users]
            user_color = self.data['user_color_indices_tensor'][batch_users]
            user_size = self.data['user_size_indices_tensor'][batch_users]
            pos_item_feat = self.data['item_features_tensor'][batch_pos_items]
            neg_item_feat = self.data['item_features_tensor'][batch_neg_items]

            user_emb = self.model.user_forward(batch_users, user_feat, user_color, user_size)
            pos_item_emb = self.model.item_forward(batch_pos_items, pos_item_feat)
            neg_item_emb = self.model.item_forward(batch_neg_items, neg_item_feat)

            temperature = getattr(self.model, 'temperature', 0.1)
            loss = self.contrastive_loss(user_emb, pos_item_emb, neg_item_emb, temperature=temperature)
            if torch.isnan(loss):
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            current_batch_size = batch_users.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        return total_loss / max(total_samples, 1)

    def _train_lightgcn_epoch(self, batch_size):
        """LightGCN 模型基于用户-正样本-负样本三元组进行训练。"""
        total_loss = 0.0
        total_samples = 0

        for user_indices, pos_item_indices in self.iterate_training_batches(batch_size):
            neg_item_indices = self.sample_negative_items_for_users(user_indices)

            user_emb, item_emb = self.model(
                self.edge_index,
                self.data['user_features_tensor'],
                self.data['user_color_indices_tensor'],
                self.data['user_size_indices_tensor'],
                self.data['item_features_tensor']
            )

            user_batch_emb = user_emb[user_indices]
            pos_item_batch_emb = item_emb[pos_item_indices]
            neg_item_batch_emb = item_emb[neg_item_indices]

            loss = self.bpr_loss(user_batch_emb, pos_item_batch_emb, neg_item_batch_emb)
            if torch.isnan(loss):
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            current_batch_size = user_indices.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        return total_loss / max(total_samples, 1)

    def _train_harn_epoch(self, batch_size):
        """HARN 模型使用 BCE 监督进行训练。"""
        total_loss = 0.0
        total_samples = 0

        for batch_users, batch_pos_items in self.iterate_training_batches(batch_size):
            batch_neg_items = self.sample_negative_items_for_users(batch_users)

            user_feat = self.data['user_features_tensor'][batch_users]
            user_color = self.data['user_color_indices_tensor'][batch_users]
            user_size = self.data['user_size_indices_tensor'][batch_users]
            pos_item_feat = self.data['item_features_tensor'][batch_pos_items]
            neg_item_feat = self.data['item_features_tensor'][batch_neg_items]

            pos_scores = self.model(
                batch_users, user_feat, user_color, user_size, batch_pos_items, pos_item_feat
            )
            neg_scores = self.model(
                batch_users, user_feat, user_color, user_size, batch_neg_items, neg_item_feat
            )

            loss = self.bce_loss(pos_scores, neg_scores)
            if torch.isnan(loss):
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            current_batch_size = batch_users.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        return total_loss / max(total_samples, 1)

    def train(self, epochs=100):
        """执行固定轮数训练，并同时保存模型参数和历史记录。"""
        print(f"\nStarting training {self.model_type}...")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")

        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('results', exist_ok=True)

        # 这些数组会被写入训练历史 CSV，供后续效率分析使用。
        training_times = []
        epoch_losses = []
        learning_rates = []

        total_start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            train_loss = self.train_epoch()
            self.scheduler.step(train_loss)

            epoch_time = time.time() - epoch_start_time
            training_times.append(epoch_time)
            epoch_losses.append(train_loss)
            learning_rates.append(self.optimizer.param_groups[0]['lr'])

            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"\nEpoch {epoch + 1}/{epochs}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Epoch Time: {epoch_time:.2f}s")
                print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

        total_training_time = time.time() - total_start_time
        avg_epoch_time = np.mean(training_times)

        # 当前实验不使用验证集早停，因此保存训练结束时的模型参数。
        checkpoint_path = f'checkpoints/best_{self.run_name}_model.pth'
        torch.save(self.model.state_dict(), checkpoint_path)

        history_path = f'results/{self.run_name}_training_history.csv'
        epoch_history_df = self._build_epoch_history(epoch_losses, training_times, learning_rates)
        epoch_history_df.to_csv(history_path, index=False)

        print("\n=== Training Summary ===")
        print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f}min)")
        print(f"Average epoch time: {avg_epoch_time:.2f}s")
        print(f"Total epochs: {len(training_times)}")
        print(f"Final model saved to {checkpoint_path}")
        print(f"Training history saved to {history_path}")

        return {
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time,
            'total_epochs': len(training_times),
            'best_ndcg': None,
            'final_train_loss': epoch_losses[-1] if epoch_losses else None,
            'training_history_path': history_path
        }

    @staticmethod
    def _build_epoch_history(epoch_losses, training_times, learning_rates):
        """将每轮训练统计整理成 DataFrame。"""
        return pd.DataFrame({
            'epoch': np.arange(1, len(epoch_losses) + 1),
            'train_loss': np.array(epoch_losses, dtype=np.float32),
            'epoch_time_seconds': np.array(training_times, dtype=np.float32),
            'learning_rate': np.array(learning_rates, dtype=np.float32)
        })

    def final_evaluation(self):
        """加载保存的模型参数，并执行最终测试集评测。"""
        print("\n=== Final Evaluation ===")

        checkpoint_path = f'checkpoints/best_{self.run_name}_model.pth'
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
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

        results_df = self.evaluator.save_results(final_results, self.run_name, inference_time)

        return {
            'results': final_results,
            'inference_time': inference_time,
            'results_df': results_df
        }
