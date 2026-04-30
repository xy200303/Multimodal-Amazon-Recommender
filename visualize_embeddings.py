from __future__ import annotations

import argparse

from regenerate_paper_figures import run_embedding_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="导出并可视化用户/商品嵌入空间。")
    parser.add_argument(
        "--model",
        default="graph_augmented_twotower",
        choices=["twotower", "graph_augmented_twotower"],
        help="需要可视化的模型名称。",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="模型 checkpoint 路径；为空时按模型名称自动推断。",
    )
    parser.add_argument(
        "--method",
        default="pca",
        choices=["pca", "tsne", "both"],
        help="降维可视化方法。",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=2,
        choices=[2, 3],
        help="可视化维度。",
    )
    parser.add_argument(
        "--item-ablation",
        default="all_features",
        help="商品特征配置，需与训练时保持一致。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="运行设备，支持 auto/cpu/cuda。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--tsne-user-sample",
        type=int,
        default=1500,
        help="t-SNE 用户采样上限。",
    )
    parser.add_argument(
        "--tsne-item-sample",
        type=int,
        default=2500,
        help="t-SNE 商品采样上限。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_embedding_visualization(
        model_name=args.model,
        checkpoint=args.checkpoint,
        method=args.method,
        dimensions=args.dimensions,
        item_ablation=args.item_ablation,
        device_name=args.device,
        seed=args.seed,
        tsne_user_sample=args.tsne_user_sample,
        tsne_item_sample=args.tsne_item_sample,
    )


if __name__ == "__main__":
    main()
