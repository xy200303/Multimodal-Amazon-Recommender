import torch
import pandas as pd
import numpy as np
import os
import time
from data_preprocessor import DataPreprocessor
from models import (CollaborativeFiltering, DeepFM, TwoTowerModel, 
                    MultiModalLightGCN, RandomBaseline, PopularityBaseline)
from trainer import Trainer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_all(device)
    
    model_configs = {
        'random_baseline': {
            'model_class': RandomBaseline,
            'params': {
                'num_items': data['num_items']
            },
            'training_config': {},
            'epochs': 0,
            'eval_every': 1
        },
        'popularity_baseline': {
            'model_class': PopularityBaseline,
            'params': {
                'num_items': data['num_items']
            },
            'training_config': {},
            'epochs': 0,
            'eval_every': 1
        },
        'collaborative_filtering': {
            'model_class': CollaborativeFiltering,
            'params': {
                'num_users': data['num_users'],
                'num_items': data['num_items'],
                'embedding_dim': 64,
                'dropout': 0.1
            },
            'training_config': {
                'lr': 0.001,
                'weight_decay': 1e-5,
                'batch_size': 4096,
                'patience': 5,
                'factor': 0.5
            },
            'epochs': 100,
            'eval_every': 10
        },
        'deepfm': {
            'model_class': DeepFM,
            'params': {
                'num_users': data['num_users'],
                'num_items': data['num_items'],
                'user_feature_dim': data['user_features_tensor'].shape[1],
                'item_feature_dim': data['item_features_tensor'].shape[1],
                'embedding_dim': 64,
                'hidden_dims': [64, 32],
                'dropout': 0.1
            },
            'training_config': {
                'lr': 0.001,
                'weight_decay': 1e-5,
                'batch_size': 4096,
                'patience': 5,
                'factor': 0.5
            },
            'epochs': 100,
            'eval_every': 10
        },
        'twotower': {
            'model_class': TwoTowerModel,
            'params': {
                'num_users': data['num_users'],
                'num_items': data['num_items'],
                'user_feature_dim': data['user_features_tensor'].shape[1],
                'item_feature_dim': data['item_features_tensor'].shape[1],
                'embedding_dim': 128,
                'hidden_dims': [256, 128],
                'dropout': 0.2
            },
            'training_config': {
                'lr': 0.001,
                'weight_decay': 1e-5,
                'batch_size': 4096,
                'patience': 5,
                'factor': 0.5
            },
            'epochs': 100,
            'eval_every': 10
        },
        'lightgcn': {
            'model_class': MultiModalLightGCN,
            'params': {
                'num_users': data['num_users'],
                'num_items': data['num_items'],
                'embedding_dim': 64,
                'num_layers': 3,
                'user_numeric_dim': 12,
                'user_vector_dim': 3,
                'num_colors': 22,
                'num_sizes': 18,
                'item_numeric_dim': 5,
                'item_vector_dim': 12,
                'dropout': 0.1
            },
            'training_config': {
                'lr': 0.0001,
                'weight_decay': 1e-4,
                'batch_size': 2048,
                'patience': 5,
                'factor': 0.5
            },
            'epochs': 100,
            'eval_every': 5
        }
    }
    
    all_results = {}
    
    for model_name, config in model_configs.items():
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*80}")
        
        model = config['model_class'](**config['params'])
        
        if model_name in ['random_baseline', 'popularity_baseline']:
            print(f"  {model_name} does not require training")
            
            if model_name == 'popularity_baseline':
                model.fit(data['interaction_matrix'])
            
            final_eval, inference_time = evaluate_baseline_model(
                model, model_name, data, device
            )
            
            all_results[model_name] = {
                'training_info': {
                    'total_training_time': 0,
                    'avg_epoch_time': 0,
                    'total_epochs': 0,
                    'best_ndcg': 0
                },
                'final_eval': final_eval,
                'config': config
            }
        else:
            trainer = Trainer(
                model=model,
                model_type=model_name,
                data=data,
                device=device,
                config=config['training_config']
            )
            
            training_info = trainer.train(
                epochs=config['epochs'],
                eval_every=config['eval_every'],
                early_stopping_patience=10
            )
            
            final_eval = trainer.final_evaluation()
            
            all_results[model_name] = {
                'training_info': training_info,
                'final_eval': final_eval,
                'config': config
            }
        
        print(f"\n✅ {model_name} training completed!")
    
    generate_comparison_report(all_results, data)

def evaluate_baseline_model(model, model_type, data, device):
    from evaluator import Evaluator
    
    evaluator = Evaluator(k_list=[5, 10, 20])
    eval_results, inference_time = evaluator.evaluate_model(
        model, model_type, data, device
    )
    
    print(f"  Baseline Evaluation:")
    print(f"    Inference Time: {inference_time:.2f}s")
    
    for k in [5, 10, 20]:
        print(f"    Top-{k}: Precision={eval_results[k]['precision']:.4f}, "
              f"Recall={eval_results[k]['recall']:.4f}, NDCG={eval_results[k]['ndcg']:.4f}")
    
    results_df = evaluator.save_results(eval_results, model_type, inference_time)
    
    return {
        'results': eval_results,
        'inference_time': inference_time
    }

def generate_comparison_report(all_results, data):
    print(f"\n{'='*80}")
    print("FINAL COMPARISON REPORT")
    print(f"{'='*80}")
    
    print(f"\nDataset Information:")
    print(f"  Users: {data['num_users']}")
    print(f"  Items: {data['num_items']}")
    print(f"  Test Users: {len(data['test_users_list'])}")
    print(f"  Test Items: {data['total_test_items']}")
    
    print(f"\n{'Model':<25} | {'Precision@10':<12} | {'Recall@10':<12} | {'NDCG@10':<12} | {'Train Time':<12} | {'Inference Time':<12}")
    print("-" * 100)
    
    comparison_data = []
    for model_name, results in all_results.items():
        final_eval = results['final_eval']
        training_info = results['training_info']
        
        if not isinstance(final_eval, dict):
            print(f"Warning: {model_name} final_eval is not a dictionary: {type(final_eval)}")
            continue
        
        results_dict = final_eval.get('results') if isinstance(final_eval.get('results'), dict) else None
        results_df = final_eval.get('results') if isinstance(final_eval.get('results'), pd.DataFrame) else None
        
        if results_dict is not None:
            precision_10 = results_dict[10]['precision']
            recall_10 = results_dict[10]['recall']
            ndcg_10 = results_dict[10]['ndcg']
        elif results_df is not None:
            row_10 = results_df[results_df['K'] == 10]
            if len(row_10) > 0:
                precision_10 = row_10['Precision'].iloc[0]
                recall_10 = row_10['Recall'].iloc[0]
                ndcg_10 = row_10['NDCG'].iloc[0]
            else:
                continue
        
        train_time = training_info['total_training_time']
        inference_time = results['final_eval']['inference_time'] if 'inference_time' in results['final_eval'] else 0
        
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Precision@10': precision_10,
            'Recall@10': recall_10,
            'NDCG@10': ndcg_10,
            'Train Time': train_time,
            'Inference Time': inference_time
        })
        
        print(f"{model_name.replace('_', ' ').title():<25} | {precision_10:>12.4f} | {recall_10:>12.4f} | {ndcg_10:>12.4f} | {train_time:>12.2f}s | {inference_time:>12.2f}s")
    
    print("-" * 100)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("\nComparison results saved to results/model_comparison.csv")
    
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    
    for model_name, results in all_results.items():
        final_eval = results['final_eval']
        training_info = results['training_info']
        
        if not isinstance(final_eval, dict):
            print(f"Warning: {model_name} final_eval is not a dictionary: {type(final_eval)}")
            continue
        
        results_dict = final_eval.get('results') if isinstance(final_eval.get('results'), dict) else None
        results_df = final_eval.get('results') if isinstance(final_eval.get('results'), pd.DataFrame) else None
        
        if results_dict is not None:
            precision_10 = results_dict[10]['precision']
            recall_10 = results_dict[10]['recall']
            ndcg_10 = results_dict[10]['ndcg']
        elif results_df is not None:
            precision_10 = results_df[results_df['K'] == 10]['Precision'].iloc[0]
            recall_10 = results_df[results_df['K'] == 10]['Recall'].iloc[0]
            ndcg_10 = results_df[results_df['K'] == 10]['NDCG'].iloc[0]
        else:
            continue
        
        print(f"\n{model_name.replace('_', ' ').upper()}:")
        print(f"  Training Summary:")
        print(f"    Total Training Time: {training_info['total_training_time']:.2f}s ({training_info['total_training_time']/60:.2f}min)")
        print(f"    Average Epoch Time: {training_info['avg_epoch_time']:.2f}s")
        print(f"    Total Epochs: {training_info['total_epochs']}")
        print(f"    Best NDCG@10: {training_info['best_ndcg']:.4f}")
        
        print(f"  Final Evaluation:")
        print(f"    Inference Time: {final_eval['inference_time']:.2f}s")
        print(f"    Total Test Items: {data['total_test_items']}")
        
        if results_dict is not None:
            print(f"    Top-5:  Precision={results_dict[5]['precision']:.4f}, "
                  f"Recall={results_dict[5]['recall']:.4f}, NDCG={results_dict[5]['ndcg']:.4f}")
            print(f"    Top-10: Precision={results_dict[10]['precision']:.4f}, "
                  f"Recall={results_dict[10]['recall']:.4f}, NDCG={results_dict[10]['ndcg']:.4f}")
            print(f"    Top-20: Precision={results_dict[20]['precision']:.4f}, "
                  f"Recall={results_dict[20]['recall']:.4f}, NDCG={results_dict[20]['ndcg']:.4f}")
        elif results_df is not None:
            print(f"    Top-5:  Precision={results_df[results_df['K'] == 5]['Precision'].iloc[0]:.4f}, "
                  f"Recall={results_df[results_df['K'] == 5]['Recall'].iloc[0]:.4f}, NDCG={results_df[results_df['K'] == 5]['NDCG'].iloc[0]:.4f}")
            print(f"    Top-10: Precision={results_df[results_df['K'] == 10]['Precision'].iloc[0]:.4f}, "
                  f"Recall={results_df[results_df['K'] == 10]['Recall'].iloc[0]:.4f}, NDCG={results_df[results_df['K'] == 10]['NDCG'].iloc[0]:.4f}")
            print(f"    Top-20: Precision={results_df[results_df['K'] == 20]['Precision'].iloc[0]:.4f}, "
                  f"Recall={results_df[results_df['K'] == 20]['Recall'].iloc[0]:.4f}, NDCG={results_df[results_df['K'] == 20]['NDCG'].iloc[0]:.4f}")
    
    print(f"\n{'='*80}")
    print("PERFORMANCE RANKING")
    print(f"{'='*80}")
    
    sorted_by_ndcg = sorted(comparison_data, key=lambda x: x['NDCG@10'], reverse=True)
    sorted_by_precision = sorted(comparison_data, key=lambda x: x['Precision@10'], reverse=True)
    sorted_by_recall = sorted(comparison_data, key=lambda x: x['Recall@10'], reverse=True)
    sorted_by_speed = sorted(comparison_data, key=lambda x: x['Train Time'])
    
    print(f"\nRanked by NDCG@10:")
    for i, model in enumerate(sorted_by_ndcg, 1):
        print(f"  {i}. {model['Model']:<25} - NDCG@10: {model['NDCG@10']:.4f}")
    
    print(f"\nRanked by Training Speed:")
    for i, model in enumerate(sorted_by_speed, 1):
        print(f"  {i}. {model['Model']:<25} - Train Time: {model['Train Time']:.2f}s")
    
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    best_ndcg_model = sorted_by_ndcg[0]
    fastest_model = sorted_by_speed[0]
    
    print(f"\n🏆 Best Performing Model: {best_ndcg_model['Model']}")
    print(f"   NDCG@10: {best_ndcg_model['NDCG@10']:.4f}")
    print(f"   Precision@10: {best_ndcg_model['Precision@10']:.4f}")
    print(f"   Recall@10: {best_ndcg_model['Recall@10']:.4f}")
    
    print(f"\n⚡ Fastest Training Model: {fastest_model['Model']}")
    print(f"   Training Time: {fastest_model['Train Time']:.2f}s")
    print(f"   NDCG@10: {fastest_model['NDCG@10']:.4f}")
    
    print(f"\n✅ All models trained and evaluated successfully!")
    print(f"📊 Results saved to results/model_comparison.csv")

if __name__ == "__main__":
    main()