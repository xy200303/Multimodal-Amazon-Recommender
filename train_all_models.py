"""统一实验入口脚本。

该脚本负责：
1. 调用数据预处理流程；
2. 构建并训练全部候选模型；
3. 在训练结束后执行统一评测；
4. 输出效果对比表和效率统计表。
"""

import argparse
import time

import pandas as pd
import torch

from data_preprocessor import DataPreprocessor, ITEM_ABLATION_MODES
from models import (
    CollaborativeFiltering,
    DeepFM,
    GraphAugmentedTwoTowerModel,
    HybridAttentionRecommendationNetwork,
    MultiModalLightGCN,
    PopularityBaseline,
    RandomBaseline,
    TwoTowerModel,
)
from trainer import Trainer

DEFAULT_MODEL_ORDER = [
    'random_baseline',
    'popularity_baseline',
    'collaborative_filtering',
    'deepfm',
    'twotower',
    'lightgcn',
    'graph_augmented_twotower',
    'harn',
]

ABLATION_MODEL_ORDER = ['twotower', 'graph_augmented_twotower']
DEFAULT_ABLATION_MODES = ['numeric_text', 'text_image', 'all_features']
ABLATION_MODE_LABELS = {
    'numeric_text': 'Numeric+Text',
    'text_image': 'Text+Image',
    'all_features': 'All Features'
}


def build_model_configs(data):
    """集中定义全部实验配置，保证实验过程可复现。"""
    return {
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
                'num_colors': data['num_colors'],
                'num_sizes': data['num_sizes'],
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
                'num_colors': data['num_colors'],
                'num_sizes': data['num_sizes'],
                'embedding_dim': 64,
                'hidden_dims': [128, 64],
                'dropout': 0.3
            },
            'training_config': {
                'lr': 0.0005,
                'weight_decay': 1e-4,
                'batch_size': 512,
                'patience': 10,
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
                'user_numeric_dim': data['user_numeric_dim'],
                'user_vector_dim': data['user_vector_dim'],
                'num_colors': data['num_colors'],
                'num_sizes': data['num_sizes'],
                'item_numeric_dim': data['item_numeric_dim'],
                'item_vector_dim': data['item_vector_dim'],
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
        },
        'graph_augmented_twotower': {
            'model_class': GraphAugmentedTwoTowerModel,
            'params': {
                'num_users': data['num_users'],
                'num_items': data['num_items'],
                'user_feature_dim': data['user_features_tensor'].shape[1],
                'item_feature_dim': data['item_features_tensor'].shape[1],
                'num_colors': data['num_colors'],
                'num_sizes': data['num_sizes'],
                'embedding_dim': 96,
                'hidden_dims': [192, 96],
                'dropout': 0.1,
                'temperature': 0.07,
                'num_graph_layers': 1
            },
            'training_config': {
                'lr': 0.001,
                'weight_decay': 1e-5,
                'batch_size': 1024,
                'patience': 8,
                'factor': 0.5
            },
            'epochs': 100,
            'eval_every': 10
        },
        'harn': {
            'model_class': HybridAttentionRecommendationNetwork,
            'params': {
                'num_users': data['num_users'],
                'num_items': data['num_items'],
                'user_feature_dim': data['user_features_tensor'].shape[1],
                'item_feature_dim': data['item_features_tensor'].shape[1],
                'num_colors': data['num_colors'],
                'num_sizes': data['num_sizes'],
                'embedding_dim': 64,
                'hidden_dims': [256, 128, 64],
                'dropout': 0.2
            },
            'training_config': {
                'lr': 0.001,
                'weight_decay': 1e-5,
                'batch_size': 1024,
                'patience': 8,
                'factor': 0.5
            },
            'epochs': 100,
            'eval_every': 10
        }
    }


def parse_args():
    """解析统一训练入口支持的命令行参数。"""
    parser = argparse.ArgumentParser(
        description='Train and evaluate recommendation models from a unified entry point.'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help='Models to run. Use space-separated names or comma-separated values, e.g. '
             '--models twotower deepfm or --models twotower,deepfm. Default: all.'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available model names and exit.'
    )
    parser.add_argument(
        '--item-ablation',
        default='all_features',
        choices=sorted(ITEM_ABLATION_MODES),
        help='Apply a single item-feature ablation mode to the whole run.'
    )
    parser.add_argument(
        '--run-ablation-study',
        action='store_true',
        help='Run the predefined item-feature ablation study on one backbone model.'
    )
    parser.add_argument(
        '--ablation-backbone',
        default='twotower',
        choices=ABLATION_MODEL_ORDER,
        help='Backbone model used for ablation study.'
    )
    parser.add_argument(
        '--ablation-modes',
        nargs='+',
        default=DEFAULT_ABLATION_MODES,
        help='Ablation modes to run. Supports space-separated or comma-separated values.'
    )
    return parser.parse_args()


def normalize_model_selection(raw_models):
    """兼容空格分隔与逗号分隔两种模型名输入方式。"""
    selected_models = []
    for value in raw_models:
        for model_name in value.split(','):
            normalized = model_name.strip().lower()
            if normalized:
                selected_models.append(normalized)
    return selected_models


def select_model_names(requested_models):
    """校验模型名称，并保持默认报告展示顺序。"""
    normalized_models = normalize_model_selection(requested_models)
    if not normalized_models or 'all' in normalized_models:
        return DEFAULT_MODEL_ORDER

    invalid_models = [name for name in normalized_models if name not in DEFAULT_MODEL_ORDER]
    if invalid_models:
        available = ', '.join(DEFAULT_MODEL_ORDER)
        invalid = ', '.join(invalid_models)
        raise ValueError(f"Unknown model name(s): {invalid}. Available models: {available}")

    selected_names = []
    for model_name in DEFAULT_MODEL_ORDER:
        if model_name in normalized_models and model_name not in selected_names:
            selected_names.append(model_name)
    return selected_names


def normalize_ablation_selection(raw_modes):
    """兼容空格与逗号两种输入方式，并校验消融模式名称。"""
    selected_modes = []
    for value in raw_modes:
        for mode_name in value.split(','):
            normalized = mode_name.strip().lower()
            if normalized:
                selected_modes.append(normalized)

    invalid_modes = [mode for mode in selected_modes if mode not in ITEM_ABLATION_MODES]
    if invalid_modes:
        available = ', '.join(sorted(ITEM_ABLATION_MODES))
        invalid = ', '.join(invalid_modes)
        raise ValueError(f"Unknown ablation mode(s): {invalid}. Available modes: {available}")

    ordered_modes = []
    for mode_name in DEFAULT_ABLATION_MODES:
        if mode_name in selected_modes and mode_name not in ordered_modes:
            ordered_modes.append(mode_name)
    return ordered_modes or DEFAULT_ABLATION_MODES


def extract_metrics_at_k(final_eval, k):
    """兼容 dict 和 DataFrame 两种评测结果格式，统一读取指定 K 的指标。"""
    if not isinstance(final_eval, dict):
        raise TypeError(f"final_eval must be a dict, got {type(final_eval)}")

    results = final_eval.get('results')
    if isinstance(results, dict):
        if k not in results:
            raise KeyError(f"K={k} not found in evaluation results.")
        return results[k]

    if isinstance(results, pd.DataFrame):
        row = results[results['K'] == k]
        if row.empty:
            raise KeyError(f"K={k} not found in evaluation DataFrame.")
        return {
            'precision': float(row['Precision'].iloc[0]),
            'recall': float(row['Recall'].iloc[0]),
            'ndcg': float(row['NDCG'].iloc[0])
        }

    results_df = final_eval.get('results_df')
    if isinstance(results_df, pd.DataFrame):
        row = results_df[results_df['K'] == k]
        if row.empty:
            raise KeyError(f"K={k} not found in evaluation DataFrame.")
        return {
            'precision': float(row['Precision'].iloc[0]),
            'recall': float(row['Recall'].iloc[0]),
            'ndcg': float(row['NDCG'].iloc[0])
        }

    raise TypeError(f"Unsupported evaluation result format: {type(results)}")


def run_single_model(model_name, config, data, device, run_name=None):
    run_name = run_name or model_name
    """完成单个模型的训练/评测，并返回效果与耗时信息。"""
    print(f"\n{'=' * 80}")
    print(f"Training {run_name.upper()}")
    print(f"{'=' * 80}")

    model_start_time = time.time()
    model = config['model_class'](**config['params'])

    # 基线模型跳过梯度训练，但仍沿用同一套评测与汇总路径。
    if model_name in ['random_baseline', 'popularity_baseline']:
        print(f"  {model_name} does not require training")

        if model_name == 'popularity_baseline':
            model.fit(data['interaction_matrix'])

        final_eval = evaluate_baseline_model(model, model_name, data, device, run_name=run_name)
        training_info = {
            'total_training_time': 0,
            'avg_epoch_time': 0,
            'total_epochs': 0,
            'best_ndcg': 0
        }
    else:
        trainer = Trainer(
            model=model,
            model_type=model_name,
            data=data,
            device=device,
            config={**config['training_config'], 'run_name': run_name}
        )
        training_info = trainer.train(epochs=config['epochs'])
        final_eval = trainer.final_evaluation()

    training_info['best_ndcg'] = extract_metrics_at_k(final_eval, 10)['ndcg']
    training_info['model_wall_time'] = time.time() - model_start_time

    print(f"\n[OK] {run_name} training completed!")
    return {
        'training_info': training_info,
        'final_eval': final_eval,
        'config': config,
        'run_name': run_name
    }


def evaluate_baseline_model(model, model_type, data, device, run_name=None):
    """在统一的全排序协议下评测无需训练的基线模型。"""
    from evaluator import Evaluator

    run_name = run_name or model_type
    evaluator = Evaluator(k_list=[5, 10, 20])
    eval_results, inference_time = evaluator.evaluate_model(model, model_type, data, device)

    print("  Baseline Evaluation:")
    print(f"    Inference Time: {inference_time:.2f}s")

    for k in [5, 10, 20]:
        print(
            f"    Top-{k}: Precision={eval_results[k]['precision']:.4f}, "
            f"Recall={eval_results[k]['recall']:.4f}, NDCG={eval_results[k]['ndcg']:.4f}"
        )

    evaluator.save_results(eval_results, run_name, inference_time)

    return {
        'results': eval_results,
        'inference_time': inference_time
    }


def generate_efficiency_report(all_results, data, preprocess_time, total_experiment_time):
    """生成按模型和按实验阶段划分的效率统计表。"""
    train_interactions = int(data['user_item_edge_index'].shape[1])
    test_users = len(data['test_users_list'])
    num_items = int(data['num_items'])
    evaluated_candidates = test_users * num_items

    efficiency_rows = []
    for model_name, results in all_results.items():
        training_info = results['training_info']
        final_eval = results['final_eval']

        train_time = float(training_info.get('total_training_time', 0))
        inference_time = float(final_eval.get('inference_time', 0))
        wall_time = float(training_info.get('model_wall_time', train_time + inference_time))
        avg_epoch_time = float(training_info.get('avg_epoch_time', 0))
        total_epochs = int(training_info.get('total_epochs', 0))
        final_train_loss = training_info.get('final_train_loss')
        history_path = training_info.get('training_history_path', '')

        efficiency_rows.append({
            'Model': model_name.replace('_', ' ').title(),
            'TrainInteractions': train_interactions,
            'TestUsers': test_users,
            'CandidateItemsPerUser': num_items,
            'TotalEvaluatedCandidates': evaluated_candidates,
            'TotalEpochs': total_epochs,
            'TrainTimeSeconds': train_time,
            'AvgEpochTimeSeconds': avg_epoch_time,
            'InferenceTimeSeconds': inference_time,
            'ModelWallTimeSeconds': wall_time,
            'TrainingInteractionsPerSecond': (train_interactions * total_epochs / train_time)
            if train_time > 0 and total_epochs > 0 else 0,
            'UsersPerSecond': (test_users / inference_time) if inference_time > 0 else 0,
            'CandidateScoresPerSecond': (evaluated_candidates / inference_time) if inference_time > 0 else 0,
            'FinalTrainLoss': final_train_loss if final_train_loss is not None else '',
            'TrainingHistoryPath': history_path
        })

    efficiency_df = pd.DataFrame(efficiency_rows)
    efficiency_df.to_csv('results/efficiency_summary.csv', index=False)

    experiment_summary_df = pd.DataFrame([
        {'Stage': 'preprocessing', 'TimeSeconds': preprocess_time},
        {'Stage': 'full_experiment', 'TimeSeconds': total_experiment_time}
    ])
    experiment_summary_df.to_csv('results/experiment_time_summary.csv', index=False)

    print(f"\n{'=' * 80}")
    print("EFFICIENCY SUMMARY")
    print(f"{'=' * 80}")
    print(f"Preprocessing Time: {preprocess_time:.2f}s")
    print(f"Total Experiment Time: {total_experiment_time:.2f}s ({total_experiment_time / 60:.2f}min)")
    print("\nPer-model efficiency results saved to results/efficiency_summary.csv")
    print("Experiment-level timing saved to results/experiment_time_summary.csv")

    if not efficiency_df.empty:
        print(
            f"\n{'Model':<25} | {'Train(s)':<10} | {'Infer(s)':<10} | "
            f"{'Wall(s)':<10} | {'Users/s':<12} | {'Scores/s':<14}"
        )
        print("-" * 95)
        for _, row in efficiency_df.iterrows():
            print(
                f"{row['Model']:<25} | {row['TrainTimeSeconds']:>10.2f} | "
                f"{row['InferenceTimeSeconds']:>10.2f} | {row['ModelWallTimeSeconds']:>10.2f} | "
                f"{row['UsersPerSecond']:>12.2f} | {row['CandidateScoresPerSecond']:>14.2f}"
            )
        print("-" * 95)


def generate_comparison_report(all_results, data, preprocess_time, total_experiment_time):
    """输出并保存主实验中使用的模型效果对比报告。"""
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON REPORT")
    print(f"{'=' * 80}")

    print("\nDataset Information:")
    print(f"  Users: {data['num_users']}")
    print(f"  Items: {data['num_items']}")
    print(f"  Test Users: {len(data['test_users_list'])}")
    print(f"  Test Items: {data['total_test_items']}")

    print(
        f"\n{'Model':<25} | {'Precision@10':<12} | {'Recall@10':<12} | "
        f"{'NDCG@10':<12} | {'Train Time':<12} | {'Inference Time':<12}"
    )
    print("-" * 100)

    comparison_data = []
    for model_name, results in all_results.items():
        final_eval = results['final_eval']
        training_info = results['training_info']

        if not isinstance(final_eval, dict):
            print(f"Warning: {model_name} final_eval is not a dictionary: {type(final_eval)}")
            continue

        metrics_10 = extract_metrics_at_k(final_eval, 10)
        precision_10 = metrics_10['precision']
        recall_10 = metrics_10['recall']
        ndcg_10 = metrics_10['ndcg']

        train_time = training_info['total_training_time']
        inference_time = final_eval['inference_time'] if 'inference_time' in final_eval else 0

        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Precision@10': precision_10,
            'Recall@10': recall_10,
            'NDCG@10': ndcg_10,
            'Train Time': train_time,
            'Inference Time': inference_time
        })

        print(
            f"{model_name.replace('_', ' ').title():<25} | {precision_10:>12.4f} | "
            f"{recall_10:>12.4f} | {ndcg_10:>12.4f} | {train_time:>12.2f}s | {inference_time:>12.2f}s"
        )

    print("-" * 100)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("\nComparison results saved to results/model_comparison.csv")

    print(f"\n{'=' * 80}")
    print("DETAILED RESULTS")
    print(f"{'=' * 80}")

    for model_name, results in all_results.items():
        final_eval = results['final_eval']
        training_info = results['training_info']

        if not isinstance(final_eval, dict):
            print(f"Warning: {model_name} final_eval is not a dictionary: {type(final_eval)}")
            continue

        metrics_5 = extract_metrics_at_k(final_eval, 5)
        metrics_10 = extract_metrics_at_k(final_eval, 10)
        metrics_20 = extract_metrics_at_k(final_eval, 20)

        print(f"\n{model_name.replace('_', ' ').upper()}:")
        print("  Training Summary:")
        print(
            f"    Total Training Time: {training_info['total_training_time']:.2f}s "
            f"({training_info['total_training_time'] / 60:.2f}min)"
        )
        print(f"    Average Epoch Time: {training_info['avg_epoch_time']:.2f}s")
        print(f"    Total Epochs: {training_info['total_epochs']}")
        print(f"    Best NDCG@10: {training_info['best_ndcg']:.4f}")

        print("  Final Evaluation:")
        print(f"    Inference Time: {final_eval['inference_time']:.2f}s")
        print(f"    Total Test Items: {data['total_test_items']}")
        print(
            f"    Top-5:  Precision={metrics_5['precision']:.4f}, "
            f"Recall={metrics_5['recall']:.4f}, NDCG={metrics_5['ndcg']:.4f}"
        )
        print(
            f"    Top-10: Precision={metrics_10['precision']:.4f}, "
            f"Recall={metrics_10['recall']:.4f}, NDCG={metrics_10['ndcg']:.4f}"
        )
        print(
            f"    Top-20: Precision={metrics_20['precision']:.4f}, "
            f"Recall={metrics_20['recall']:.4f}, NDCG={metrics_20['ndcg']:.4f}"
        )

    print(f"\n{'=' * 80}")
    print("PERFORMANCE RANKING")
    print(f"{'=' * 80}")

    sorted_by_ndcg = sorted(comparison_data, key=lambda x: x['NDCG@10'], reverse=True)
    sorted_by_speed = sorted(comparison_data, key=lambda x: x['Train Time'])

    print("\nRanked by NDCG@10:")
    for i, model in enumerate(sorted_by_ndcg, 1):
        print(f"  {i}. {model['Model']:<25} - NDCG@10: {model['NDCG@10']:.4f}")

    print("\nRanked by Training Speed:")
    for i, model in enumerate(sorted_by_speed, 1):
        print(f"  {i}. {model['Model']:<25} - Train Time: {model['Train Time']:.2f}s")

    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print(f"{'=' * 80}")

    best_ndcg_model = sorted_by_ndcg[0]
    fastest_model = sorted_by_speed[0]

    print(f"\n[BEST] Best Performing Model: {best_ndcg_model['Model']}")
    print(f"   NDCG@10: {best_ndcg_model['NDCG@10']:.4f}")
    print(f"   Precision@10: {best_ndcg_model['Precision@10']:.4f}")
    print(f"   Recall@10: {best_ndcg_model['Recall@10']:.4f}")

    print(f"\n[FAST] Fastest Training Model: {fastest_model['Model']}")
    print(f"   Training Time: {fastest_model['Train Time']:.2f}s")
    print(f"   NDCG@10: {fastest_model['NDCG@10']:.4f}")

    generate_efficiency_report(all_results, data, preprocess_time, total_experiment_time)

    print("\n[OK] All models trained and evaluated successfully!")
    print("[INFO] Results saved to results/model_comparison.csv")


def generate_ablation_report(backbone_name, ablation_results, preprocess_seconds):
    """输出并保存模态消融实验汇总表。"""
    print(f"\n{'=' * 80}")
    print(f"ABLATION REPORT - {backbone_name.upper()}")
    print(f"{'=' * 80}")

    report_rows = []
    for ablation_mode, results in ablation_results.items():
        final_eval = results['final_eval']
        training_info = results['training_info']
        metrics_5 = extract_metrics_at_k(final_eval, 5)
        metrics_10 = extract_metrics_at_k(final_eval, 10)
        metrics_20 = extract_metrics_at_k(final_eval, 20)

        report_rows.append({
            'Backbone': backbone_name,
            'AblationMode': ablation_mode,
            'ModeLabel': ABLATION_MODE_LABELS.get(ablation_mode, ablation_mode),
            'Precision@5': metrics_5['precision'],
            'Recall@5': metrics_5['recall'],
            'NDCG@5': metrics_5['ndcg'],
            'Precision@10': metrics_10['precision'],
            'Recall@10': metrics_10['recall'],
            'NDCG@10': metrics_10['ndcg'],
            'Precision@20': metrics_20['precision'],
            'Recall@20': metrics_20['recall'],
            'NDCG@20': metrics_20['ndcg'],
            'TrainTimeSeconds': training_info['total_training_time'],
            'InferenceTimeSeconds': final_eval.get('inference_time', 0),
            'PreprocessSeconds': preprocess_seconds[ablation_mode],
            'RunName': results.get('run_name', f'{backbone_name}_{ablation_mode}')
        })

    ablation_df = pd.DataFrame(report_rows)
    output_path = f'results/{backbone_name}_ablation_comparison.csv'
    ablation_df.to_csv(output_path, index=False)

    print(
        f"\n{'Mode':<18} | {'P@10':<8} | {'R@10':<8} | {'NDCG@10':<8} | "
        f"{'Train(s)':<10} | {'Infer(s)':<10}"
    )
    print("-" * 80)
    for _, row in ablation_df.iterrows():
        print(
            f"{row['ModeLabel']:<18} | {row['Precision@10']:>8.4f} | "
            f"{row['Recall@10']:>8.4f} | {row['NDCG@10']:>8.4f} | "
            f"{row['TrainTimeSeconds']:>10.2f} | {row['InferenceTimeSeconds']:>10.2f}"
        )
    print("-" * 80)
    print(f"Ablation results saved to {output_path}")


def main():
    """执行选定实验，并输出完整对比报告。"""
    experiment_start_time = time.time()
    args = parse_args()

    if args.list_models:
        print("Available models:")
        for model_name in DEFAULT_MODEL_ORDER:
            print(f"  - {model_name}")
        return

    if args.run_ablation_study:
        selected_ablation_modes = normalize_ablation_selection(args.ablation_modes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print(f"Ablation backbone: {args.ablation_backbone}")
        print(f"Ablation modes: {', '.join(selected_ablation_modes)}")

        ablation_results = {}
        preprocess_seconds = {}
        for ablation_mode in selected_ablation_modes:
            print(f"\nRunning ablation mode: {ablation_mode}")
            preprocess_start_time = time.time()
            preprocessor = DataPreprocessor(ablation_mode=ablation_mode)
            data = preprocessor.preprocess_all(device)
            preprocess_seconds[ablation_mode] = time.time() - preprocess_start_time

            config = build_model_configs(data)[args.ablation_backbone]
            run_name = f"{args.ablation_backbone}_{ablation_mode}"
            ablation_results[ablation_mode] = run_single_model(
                args.ablation_backbone,
                config,
                data,
                device,
                run_name=run_name
            )

        generate_ablation_report(args.ablation_backbone, ablation_results, preprocess_seconds)
        return

    selected_model_names = select_model_names(args.models)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Selected models: {', '.join(selected_model_names)}")
    print(f"Item feature ablation mode: {args.item_ablation}")

    # 预处理耗时单独统计，便于论文中的效率分析。
    preprocess_start_time = time.time()
    preprocessor = DataPreprocessor(ablation_mode=args.item_ablation)
    data = preprocessor.preprocess_all(device)
    preprocess_time = time.time() - preprocess_start_time

    all_model_configs = build_model_configs(data)
    model_configs = {name: all_model_configs[name] for name in selected_model_names}

    all_results = {}
    for model_name, config in model_configs.items():
        run_name = model_name
        if args.item_ablation != 'all_features':
            run_name = f"{model_name}_{args.item_ablation}"
        all_results[model_name] = run_single_model(model_name, config, data, device, run_name=run_name)

    total_experiment_time = time.time() - experiment_start_time
    generate_comparison_report(all_results, data, preprocess_time, total_experiment_time)


if __name__ == "__main__":
    main()
