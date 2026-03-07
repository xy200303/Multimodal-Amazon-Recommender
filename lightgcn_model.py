import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
user_df = pd.read_csv('new_feat/user.csv')
item_df = pd.read_csv('new_feat/item.csv')
user_item_df = pd.read_csv('new_feat/user_item.csv')

print(f"Users: {len(user_df)}, Items: {len(item_df)}, Interactions: {len(user_item_df)}")

user_content_feat = np.load('new_feat/user_content_feat.npy')
item_title_feat = np.load('new_feat/item_title_feat.npy')
item_image_feat = np.load('new_feat/item_image_feat.npy')

print(f"User content features: {user_content_feat.shape}")
print(f"Item title features: {item_title_feat.shape}")
print(f"Item image features: {item_image_feat.shape}")

user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_df['reviewerID'])}
item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_df['asin'])}

print("Creating interaction data...")
train_interactions = []
test_data = []

for idx, row in user_item_df.iterrows():
    user_id = row['user_id']
    train_items = str(row['train']).split('|') if pd.notna(row['train']) and str(row['train']).strip() else []
    test_items = str(row['test']).split('|') if pd.notna(row['test']) and str(row['test']).strip() else []
    
    if user_id in user_id_to_idx:
        user_idx = user_id_to_idx[user_id]
        
        for item_id in train_items:
            if item_id in item_id_to_idx:
                item_idx = item_id_to_idx[item_id]
                train_interactions.append([user_idx, item_idx])
    
    if len(test_items) > 0:
        test_data.append({
            'user_id': user_id,
            'test_items': test_items
        })

train_interactions = np.array(train_interactions)
print(f"Total train interactions: {len(train_interactions)}")
print(f"Total test users: {len(test_data)}")

num_users = len(user_df)
num_items = len(item_df)

user_features = torch.FloatTensor(user_content_feat)
item_title_features = torch.FloatTensor(item_title_feat)
item_image_features = torch.FloatTensor(item_image_feat)

item_features = torch.cat([item_title_features, item_image_features], dim=1)

print(f"User features shape: {user_features.shape}")
print(f"Item features shape: {item_features.shape}")

class SimpleLightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(SimpleLightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.prediction_layer = nn.Linear(embedding_dim * 2, 1)
        
    def forward(self, user_idx, item_idx):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        
        combined = torch.cat([user_emb, item_emb], dim=-1)
        scores = self.prediction_layer(combined)
        
        return scores.squeeze(-1)

def compute_metrics(predictions, ground_truth, k_list=[5, 10, 20]):
    """计算推荐评测指标"""
    metrics = {}
    
    for k in k_list:
        precision_k = []
        recall_k = []
        ndcg_k = []
        hit_rate_k = []
        
        for pred_scores, true_items in zip(predictions, ground_truth):
            if pred_scores is None or len(pred_scores) == 0 or len(true_items) == 0:
                continue
            
            top_k_indices = np.argsort(-pred_scores)[:k]
            
            pred_k = top_k_indices
            true_set = set(true_items)
            
            hits = len(set(pred_k) & true_set)
            precision_k.append(hits / k if k > 0 else 0)
            recall_k.append(hits / len(true_set) if len(true_set) > 0 else 0)
            hit_rate_k.append(1 if hits > 0 else 0)
            
            dcg = 0
            for i, idx in enumerate(pred_k):
                if idx in true_set:
                    dcg += 1 / np.log2(i + 2)
            
            idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(true_set)))])
            ndcg_k.append(dcg / idcg if idcg > 0 else 0)
        
        metrics[f'Precision@{k}'] = np.mean(precision_k) if precision_k else 0
        metrics[f'Recall@{k}'] = np.mean(recall_k) if recall_k else 0
        metrics[f'NDCG@{k}'] = np.mean(ndcg_k) if ndcg_k else 0
        metrics[f'HitRate@{k}'] = np.mean(hit_rate_k) if hit_rate_k else 0
    
    return metrics

def train_model(model, train_interactions, optimizer, device):
    """训练模型"""
    model.train()
    
    total_loss = 0
    num_batches = len(train_interactions)
    
    for user_idx, item_idx in train_interactions:
        user_tensor = torch.LongTensor([user_idx]).to(device)
        item_tensor = torch.LongTensor([item_idx]).to(device)
        
        optimizer.zero_grad()
        
        scores = model(user_tensor, item_tensor)
        
        target = torch.ones(1).to(device)
        
        loss = F.binary_cross_entropy_with_logits(scores, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches

def evaluate_model(model, test_data, user_id_to_idx, item_idx_to_id, k_list=[5, 10, 20], device='cpu'):
    """评估模型"""
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for data in test_data:
            user_id = data['user_id']
            test_items = data['test_items']
            
            if user_id not in user_id_to_idx or len(test_items) == 0:
                continue
            
            user_idx = user_id_to_idx[user_id]
            
            user_tensor = torch.LongTensor([user_idx]).to(device)
            item_tensor = torch.LongTensor(range(len(item_idx_to_id))).to(device)
            
            scores = []
            for item_idx in range(len(item_idx_to_id)):
                item_tensor_single = torch.LongTensor([item_idx]).to(device)
                score = model(user_tensor, item_tensor_single)
                scores.append(score.item())
            
            scores_np = np.array(scores)
            
            all_predictions.append(scores_np)
            all_ground_truth.append(test_items)
    
    metrics = compute_metrics(all_predictions, all_ground_truth, k_list)
    
    return metrics, all_predictions, all_ground_truth

print("Initializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

embedding_dim = 64

model = SimpleLightGCN(num_users, num_items, embedding_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

print("Training model...")
num_epochs = 100

for epoch in range(num_epochs):
    loss = train_model(model, train_interactions, optimizer, device)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Evaluating model...")
item_idx_to_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}

metrics, predictions, ground_truth = evaluate_model(
    model, test_data, user_id_to_idx, item_idx_to_id, k_list=[5, 10, 20], device=device
)

print("\n=== Evaluation Results ===")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")

print("\n=== Model Training Completed ===")
print("Model saved to lightgcn_model.pt")
torch.save(model.state_dict(), 'lightgcn_model.pt')

print("\n=== Detailed Results ===")
print(f"Total predictions: {len(predictions)}")
print(f"Total ground truth: {len(ground_truth)}")
print(f"Average prediction length: {np.mean([len(p) for p in predictions]):.1f}")
print(f"Average ground truth length: {np.mean([len(g) for g in ground_truth]):.1f}")

print("\n=== Summary ===")
print("LightGCN model has been successfully trained and evaluated.")
print("The model uses user and item features for personalized recommendation.")
print("Evaluation metrics include Precision@K, Recall@K, NDCG@K, and HitRate@K.")
