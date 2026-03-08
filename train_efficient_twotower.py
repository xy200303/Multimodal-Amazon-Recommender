import torch
import pandas as pd
import numpy as np
import os
import time
from data_preprocessor import DataPreprocessor
from models import EfficientTwoTowerModel
from trainer import Trainer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_all(device)
    
    print(f"\n{'='*80}")
    print("EFFICIENT TWO-TOWER MODEL TRAINING")
    print(f"{'='*80}")
    
    efficient_twotower_config = {
        'model_class': EfficientTwoTowerModel,
        'params': {
            'num_users': data['num_users'],
            'num_items': data['num_items'],
            'user_feature_dim': data['user_features_tensor'].shape[1],
            'item_feature_dim': data['item_features_tensor'].shape[1],
            'embedding_dim': 96,
            'hidden_dims': [192, 96],
            'dropout': 0.1,
            'temperature': 0.07
        },
        'training_config': {
            'lr': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 4096,
            'patience': 5,
            'factor': 0.5
        },
        'epochs': 100,
        'eval_every': 5
    }
    
    print(f"\nEfficient Two-Tower Model Configuration:")
    print(f"  Embedding Dimension: {efficient_twotower_config['params']['embedding_dim']}")
    print(f"  Hidden Dimensions: {efficient_twotower_config['params']['hidden_dims']}")
    print(f"  Dropout: {efficient_twotower_config['params']['dropout']}")
    print(f"  Temperature: {efficient_twotower_config['params']['temperature']}")
    print(f"  Learning Rate: {efficient_twotower_config['training_config']['lr']}")
    print(f"  Weight Decay: {efficient_twotower_config['training_config']['weight_decay']}")
    print(f"  Batch Size: {efficient_twotower_config['training_config']['batch_size']}")
    
    model = efficient_twotower_config['model_class'](**efficient_twotower_config['params'])
    
    print(f"\nModel Architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    trainer = Trainer(
        model=model,
        model_type='efficient_twotower',
        data=data,
        device=device,
        config=efficient_twotower_config['training_config']
    )
    
    training_info = trainer.train(
        epochs=efficient_twotower_config['epochs'],
        eval_every=efficient_twotower_config['eval_every'],
        early_stopping_patience=10
    )
    
    final_eval = trainer.final_evaluation()
    
    print(f"\n{'='*80}")
    print("EFFICIENT TWO-TOWER MODEL FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\nTraining Summary:")
    print(f"  Total Training Time: {training_info['total_training_time']:.2f}s ({training_info['total_training_time']/60:.2f}min)")
    print(f"  Average Epoch Time: {training_info['avg_epoch_time']:.2f}s")
    print(f"  Total Epochs: {training_info['total_epochs']}")
    print(f"  Best NDCG@10: {training_info['best_ndcg']:.4f}")
    
    print(f"\nFinal Evaluation:")
    print(f"  Inference Time: {final_eval['inference_time']:.2f}s")
    print(f"  Total Test Items: {data['total_test_items']}")
    
    for k in [5, 10, 20]:
        print(f"  Top-{k}:")
        print(f"    Precision@{k}: {final_eval['results'][k]['precision']:.4f}")
        print(f"    Recall@{k}: {final_eval['results'][k]['recall']:.4f}")
        print(f"    NDCG@{k}: {final_eval['results'][k]['ndcg']:.4f}")
    
    results = {
        'model_name': 'Efficient Two-Tower',
        'training_info': training_info,
        'final_eval': final_eval
    }
    
    return results

if __name__ == "__main__":
    results = main()
    print(f"\n✅ Efficient Two-Tower model training completed!")
    print(f"📊 Best NDCG@10: {results['training_info']['best_ndcg']:.4f}")
    print(f"⚡ Training Time: {results['training_info']['total_training_time']:.2f}s")
    print(f"🚀 Inference Time: {results['final_eval']['inference_time']:.2f}s")