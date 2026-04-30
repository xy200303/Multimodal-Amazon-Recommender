from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data_preprocessor import DataPreprocessor, ITEM_ABLATION_MODES
from train_all_models import build_model_configs


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
MD_PATH = next(path for path in ROOT.glob("*.md") if "完整" in path.stem)

FIGURE_SEQUENCE = [
    "paper_dataset_yearly_review_trend.jpg",
    "paper_dataset_rating_distribution_share.jpg",
    "paper_review_text_length_histogram.jpg",
    "paper_user_activity_distribution.jpg",
    "paper_user_avg_review_text_length_distribution.jpg",
    "paper_user_brand_preference_top15.jpg",
    "paper_item_metadata_feature_coverage.jpg",
    "paper_item_image_count_distribution.jpg",
    "paper_item_brand_top20_share.jpg",
    "paper_image_vector_pca.jpg",
    "paper_text_vector_pca.jpg",
    "paper_joint_embedding_tsne_3d.jpg",
    "paper_user_embedding_tsne_3d.jpg",
    "paper_item_embedding_tsne_3d.jpg",
]


def viridis_colors(count: int, start: float = 0.12, end: float = 0.92) -> np.ndarray:
    if count <= 1:
        return np.asarray([plt.cm.viridis(0.62)])
    return plt.cm.viridis(np.linspace(start, end, count))


def style_axes(ax: plt.Axes, x_grid: bool = False, y_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B0B8C1")
    ax.spines["bottom"].set_color("#B0B8C1")
    ax.tick_params(axis="both", labelsize=11)
    if y_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.22)
    if x_grid:
        ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.18)


def count_space_separated_urls(value: object) -> int:
    if pd.isna(value):
        return 0
    return len([token for token in str(value).split() if token.strip()])


def parse_vector(value: object) -> np.ndarray:
    if pd.isna(value):
        return np.zeros(64, dtype=np.float32)
    text = str(value).strip()
    if not text:
        return np.zeros(64, dtype=np.float32)
    try:
        parsed = ast.literal_eval(text)
        return np.asarray(parsed, dtype=np.float32)
    except (ValueError, SyntaxError):
        cleaned = text.strip("[]")
        values = np.fromstring(cleaned, sep=",", dtype=np.float32)
        if values.size == 0:
            values = np.fromstring(cleaned, sep=" ", dtype=np.float32)
        return values


def load_dataset_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reviews_df = pd.read_csv(ROOT / "new_dataset" / "reviews.csv")
    user_feat_df = pd.read_csv(ROOT / "new_feat" / "user.csv")
    item_df = pd.read_csv(ROOT / "new_dataset" / "item.csv")
    item_feat_df = pd.read_csv(ROOT / "new_feat" / "item.csv")
    return reviews_df, user_feat_df, item_df, item_feat_df


def save_current_figure_as_jpg(src_name: str, dst_name: str) -> None:
    src_path = FIGURE_DIR / src_name
    dst_path = FIGURE_DIR / dst_name
    from PIL import Image

    image = Image.open(src_path).convert("RGB")
    image.save(dst_path, quality=95)


def plot_yearly_review_trend(reviews_df: pd.DataFrame, dst_name: str) -> None:
    review_dates = pd.to_datetime(reviews_df["unixReviewTime"], unit="s", errors="coerce")
    yearly_counts = review_dates.dt.year.value_counts().sort_index()
    colors = viridis_colors(len(yearly_counts))

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(yearly_counts.index, yearly_counts.values, linewidth=2.8, color=colors[-2], zorder=2)
    ax.fill_between(yearly_counts.index, yearly_counts.values, color=colors[-2], alpha=0.12, zorder=1)
    ax.scatter(yearly_counts.index, yearly_counts.values, s=90, c=colors, edgecolors="white", linewidths=1.2, zorder=3)
    ax.set_xlabel("年份")
    ax.set_ylabel("评论数量")
    style_axes(ax)

    for year, count in yearly_counts.items():
        ax.text(year, count, str(int(count)), fontsize=9, ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / dst_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rating_share(reviews_df: pd.DataFrame, dst_name: str) -> None:
    rating_counts = reviews_df["overall"].value_counts().sort_index()
    labels = [f"{rating:.0f}星" for rating in rating_counts.index]
    colors = viridis_colors(len(labels), start=0.18, end=0.88)

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    _, _, autotexts = ax.pie(
        rating_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.2},
        colors=colors,
        textprops={"fontsize": 11},
    )
    for text in autotexts:
        text.set_color("white")
        text.set_fontsize(10)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / dst_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_review_text_length_histogram(reviews_df: pd.DataFrame, dst_name: str) -> None:
    text_lengths = reviews_df["reviewText"].fillna("").astype(str).str.len()
    clipped = text_lengths.clip(upper=text_lengths.quantile(0.99))
    counts, bin_edges = np.histogram(clipped, bins=50)
    colors = viridis_colors(len(counts))

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        align="edge",
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.96,
    )
    ax.axvline(text_lengths.mean(), color="#2A9D8F", linestyle="--", linewidth=2.2, label=f"均值：{text_lengths.mean():.1f}")
    ax.axvline(text_lengths.median(), color="#264653", linestyle=":", linewidth=2.2, label=f"中位数：{text_lengths.median():.1f}")
    ax.set_xlabel("评论文本长度")
    ax.set_ylabel("评论数量")
    ax.legend(frameon=True, fontsize=10)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / dst_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_user_activity_distribution(user_feat_df: pd.DataFrame, dst_name: str) -> None:
    counts = user_feat_df["review_count"].fillna(0)
    bins = np.arange(0.5, min(int(counts.max()), 30) + 1.5, 1.0)
    hist_counts, bin_edges = np.histogram(counts.clip(upper=30), bins=bins)
    colors = viridis_colors(len(hist_counts))

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(
        bin_edges[:-1],
        hist_counts,
        width=np.diff(bin_edges),
        align="edge",
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.96,
    )
    ax.set_xlabel("用户评论数")
    ax.set_ylabel("用户数量")
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / dst_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_user_avg_review_text_length_distribution(user_feat_df: pd.DataFrame, dst_name: str) -> None:
    avg_lengths = user_feat_df["avg_text_length"].fillna(0)
    clipped = avg_lengths.clip(upper=avg_lengths.quantile(0.98))
    counts, bin_edges = np.histogram(clipped, bins=40)
    colors = viridis_colors(len(counts))

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        align="edge",
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.96,
    )
    ax.axvline(avg_lengths.mean(), color="#1D3557", linestyle="--", linewidth=2.2, label=f"均值：{avg_lengths.mean():.1f}")
    ax.axvline(avg_lengths.median(), color="#457B9D", linestyle=":", linewidth=2.2, label=f"中位数：{avg_lengths.median():.1f}")
    ax.set_xlabel("用户平均评论文本长度")
    ax.set_ylabel("用户数量")
    ax.legend(frameon=True, fontsize=10)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / dst_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_user_brand_preference_top15(user_feat_df: pd.DataFrame, dst_name: str) -> None:
    brand_counts = user_feat_df["top_category"].fillna("").astype(str).str.strip()
    brand_counts = brand_counts[brand_counts.ne("")].value_counts().head(15).sort_values()
    colors = viridis_colors(len(brand_counts))

    fig, ax = plt.subplots(figsize=(10, 6.4))
    bars = ax.barh(brand_counts.index, brand_counts.values, color=colors, edgecolor="none")
    ax.set_xlabel("用户数量")
    style_axes(ax, x_grid=True, y_grid=False)

    for bar, value in zip(bars, brand_counts.values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {int(value)}", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / dst_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_item_metadata_feature_coverage(item_df: pd.DataFrame, dst_name: str) -> None:
    coverage = {
        "品牌信息": item_df["brand"].fillna("").astype(str).str.strip().ne("").mean() * 100,
        "价格信息": item_df["price"].fillna("").astype(str).str.strip().ne("").mean() * 100,
        "属性说明": item_df["feature"].fillna("").astype(str).str.strip().ne("").mean() * 100,
        "描述文本": item_df["description"].fillna("").astype(str).str.strip().ne("").mean() * 100,
        "图片信息": item_df["imageURLHighRes"].apply(count_space_separated_urls).gt(0).mean() * 100,
    }
    colors = viridis_colors(len(coverage))

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    bars = ax.bar(coverage.keys(), coverage.values(), color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("覆盖率（%）")
    style_axes(ax)

    for bar, value in zip(bars, coverage.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / dst_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_item_image_count_distribution(item_df: pd.DataFrame, dst_name: str) -> None:
    image_counts = item_df["imageURLHighRes"].apply(count_space_separated_urls)
    bins = np.arange(-0.5, min(int(image_counts.max()), 10) + 1.5, 1.0)
    counts, bin_edges = np.histogram(image_counts.clip(upper=10), bins=bins)
    colors = viridis_colors(len(counts))

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        align="edge",
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.96,
    )
    ax.set_xlabel("商品图片数量")
    ax.set_ylabel("商品数量")
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / dst_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_item_brand_top20_share(item_df: pd.DataFrame, dst_name: str) -> None:
    brand_counts = item_df["brand"].fillna("").astype(str).str.strip()
    brand_counts = brand_counts[brand_counts.ne("")].value_counts().head(20)
    shares = brand_counts / brand_counts.sum() * 100
    colors = viridis_colors(len(shares))

    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.bar(np.arange(len(shares)), shares.values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(np.arange(len(shares)))
    ax.set_xticklabels(shares.index, rotation=45, ha="right")
    ax.set_ylabel("占比（%）")
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / dst_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_item_vector_projection(item_feat_df: pd.DataFrame, item_popularity: pd.Series, vector_column: str, dst_name: str) -> None:
    vectors = np.vstack(item_feat_df[vector_column].map(parse_vector).to_list())
    valid_mask = np.linalg.norm(vectors, axis=1) > 0
    valid_vectors = vectors[valid_mask]

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(valid_vectors)
    popularity = item_feat_df.loc[valid_mask, "asin"].map(item_popularity).fillna(0)
    color_values = np.log1p(popularity.to_numpy())

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=color_values, cmap="viridis", s=12, alpha=0.78, edgecolors="none")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("样本热度对数值", fontsize=11)
    colorbar.ax.tick_params(labelsize=10)
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    style_axes(ax, x_grid=True, y_grid=True)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / dst_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_checkpoint_path(model_name: str, checkpoint_path: str) -> str:
    if checkpoint_path:
        return checkpoint_path
    return os.path.join("checkpoints", f"best_{model_name}_model.pth")


def prepare_embedding_data(device: torch.device, item_ablation: str):
    preprocessor = DataPreprocessor(ablation_mode=item_ablation)
    data = preprocessor.preprocess_all(device)
    ordered_user_df = preprocessor.user_df.sort_values("user_idx").reset_index(drop=True).copy()
    ordered_item_df = preprocessor.item_df.sort_values("item_idx").reset_index(drop=True).copy()
    return preprocessor, data, ordered_user_df, ordered_item_df


def build_and_load_model(model_name: str, data: dict, device: torch.device, checkpoint_path: str):
    model_configs = build_model_configs(data)
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model for visualization: {model_name}")

    model_config = model_configs[model_name]
    model = model_config["model_class"](**model_config["params"]).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_model_embeddings(model, model_name: str, data: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        if model_name == "graph_augmented_twotower" and hasattr(model, "refresh_graph_cache"):
            model.refresh_graph_cache(
                data["edge_index"],
                data["user_features_tensor"],
                data["user_color_indices_tensor"],
                data["user_size_indices_tensor"],
                data["item_features_tensor"],
            )

        all_user_indices = torch.arange(data["num_users"], device=device, dtype=torch.long)
        all_item_indices = torch.arange(data["num_items"], device=device, dtype=torch.long)

        user_vectors = model.user_forward(
            all_user_indices,
            data["user_features_tensor"],
            data["user_color_indices_tensor"],
            data["user_size_indices_tensor"],
        )
        item_vectors = model.item_forward(all_item_indices, data["item_features_tensor"])

        user_vectors = F.normalize(user_vectors, p=2, dim=-1).cpu().numpy()
        item_vectors = F.normalize(item_vectors, p=2, dim=-1).cpu().numpy()

        if hasattr(model, "clear_graph_cache"):
            model.clear_graph_cache()

    return user_vectors, item_vectors


def get_primary_token(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return "unknown"


def build_embedding_metadata(preprocessor, data: dict, user_df: pd.DataFrame, item_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_metadata = user_df.copy()
    item_metadata = item_df.copy()

    user_metadata["dominant_color"] = user_metadata["top_style_colors"].apply(get_primary_token)
    user_metadata["dominant_size"] = user_metadata["top_style_sizes"].apply(get_primary_token)
    user_metadata["interaction_degree"] = data["interaction_matrix"].sum(axis=1).astype(int)

    item_metadata["price_numeric"] = item_metadata["price"].apply(preprocessor.extract_price)
    item_metadata["interaction_degree"] = data["interaction_matrix"].sum(axis=0).astype(int)
    item_metadata["has_valid_price"] = item_metadata["price_numeric"].gt(0).astype(int)
    return user_metadata, item_metadata


def run_joint_pca(user_vectors: np.ndarray, item_vectors: np.ndarray, seed: int, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_vectors = np.vstack([user_vectors, item_vectors])
    pca = PCA(n_components=n_components, random_state=seed)
    all_coords = pca.fit_transform(all_vectors)
    user_count = user_vectors.shape[0]
    return all_coords[:user_count], all_coords[user_count:], pca.explained_variance_ratio_


def sample_for_tsne(vectors: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    total_points = vectors.shape[0]
    if total_points <= max_points:
        return np.arange(total_points), vectors
    rng = np.random.default_rng(seed)
    sampled_indices = np.sort(rng.choice(total_points, size=max_points, replace=False))
    return sampled_indices, vectors[sampled_indices]


def run_tsne(vectors: np.ndarray, max_points: int, seed: int, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    sampled_indices, sampled_vectors = sample_for_tsne(vectors, max_points=max_points, seed=seed)
    sample_size = sampled_vectors.shape[0]
    perplexity = min(30, max(5, sample_size // 20))
    perplexity = min(perplexity, sample_size - 1)

    reducer = TSNE(
        n_components=n_components,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    )
    return sampled_indices, reducer.fit_transform(sampled_vectors)


def build_output_suffix(method_name: str, n_components: int) -> str:
    if n_components == 2:
        return method_name
    return f"{method_name}_{n_components}d"


def save_embedding_csv(metadata_df: pd.DataFrame, coords: np.ndarray, output_path: str, prefix: str) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    export_df = metadata_df.copy()
    export_df[f"{prefix}_x"] = coords[:, 0]
    export_df[f"{prefix}_y"] = coords[:, 1]
    if coords.shape[1] >= 3:
        export_df[f"{prefix}_z"] = coords[:, 2]
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_user_embeddings(user_df: pd.DataFrame, coords: np.ndarray, output_path: str) -> None:
    is_3d = coords.shape[1] == 3
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d" if is_3d else None)
    color_values = np.log1p(user_df["review_count"].fillna(0).to_numpy())

    if is_3d:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color_values, cmap="viridis", s=16, alpha=0.72, edgecolors="none")
        ax.set_zlabel("主成分 3")
        ax.view_init(elev=22, azim=38)
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=color_values, cmap="viridis", s=20, alpha=0.78, edgecolors="none")

    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.grid(alpha=0.15, linewidth=0.6)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("用户活跃度对数值")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_item_embeddings(item_df: pd.DataFrame, coords: np.ndarray, output_path: str) -> None:
    is_3d = coords.shape[1] == 3
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d" if is_3d else None)
    color_values = np.log1p(item_df["interaction_degree"].fillna(0).to_numpy())

    if is_3d:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color_values, cmap="viridis", s=12, alpha=0.65, edgecolors="none")
        ax.set_zlabel("主成分 3")
        ax.view_init(elev=22, azim=38)
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=color_values, cmap="viridis", s=15, alpha=0.72, edgecolors="none")

    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.grid(alpha=0.15, linewidth=0.6)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("商品热度对数值")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_joint_embeddings(user_coords: np.ndarray, item_coords: np.ndarray, output_path: str) -> None:
    is_3d = user_coords.shape[1] == 3
    fig = plt.figure(figsize=(8.8, 6.8))
    ax = fig.add_subplot(111, projection="3d" if is_3d else None)
    user_color = plt.cm.viridis(0.78)
    item_color = plt.cm.viridis(0.22)

    if is_3d:
        ax.scatter(item_coords[:, 0], item_coords[:, 1], item_coords[:, 2], c=[item_color], s=10, alpha=0.28, label="商品", edgecolors="none")
        ax.scatter(user_coords[:, 0], user_coords[:, 1], user_coords[:, 2], c=[user_color], s=16, alpha=0.68, label="用户", edgecolors="none")
        ax.set_zlabel("主成分 3")
        ax.view_init(elev=20, azim=42)
    else:
        ax.scatter(item_coords[:, 0], item_coords[:, 1], c=[item_color], s=12, alpha=0.36, label="商品", edgecolors="none")
        ax.scatter(user_coords[:, 0], user_coords[:, 1], c=[user_color], s=18, alpha=0.74, label="用户", edgecolors="none")

    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.grid(alpha=0.15, linewidth=0.6)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_pca_outputs(model_name: str, seed: int, user_vectors: np.ndarray, item_vectors: np.ndarray, user_metadata: pd.DataFrame, item_metadata: pd.DataFrame, n_components: int) -> np.ndarray:
    user_coords, item_coords, explained = run_joint_pca(user_vectors, item_vectors, seed=seed, n_components=n_components)
    output_suffix = build_output_suffix("pca", n_components)

    save_embedding_csv(user_metadata, user_coords, str(RESULT_DIR / f"{model_name}_user_embedding_{output_suffix}.csv"), prefix=output_suffix)
    save_embedding_csv(item_metadata, item_coords, str(RESULT_DIR / f"{model_name}_item_embedding_{output_suffix}.csv"), prefix=output_suffix)

    plot_user_embeddings(user_metadata, user_coords, str(FIGURE_DIR / f"{model_name}_user_embedding_{output_suffix}.png"))
    plot_item_embeddings(item_metadata, item_coords, str(FIGURE_DIR / f"{model_name}_item_embedding_{output_suffix}.png"))
    plot_joint_embeddings(user_coords, item_coords, str(FIGURE_DIR / f"{model_name}_joint_embedding_{output_suffix}.png"))
    return explained


def export_tsne_outputs(model_name: str, seed: int, user_vectors: np.ndarray, item_vectors: np.ndarray, user_metadata: pd.DataFrame, item_metadata: pd.DataFrame, tsne_user_sample: int, tsne_item_sample: int, n_components: int) -> tuple[int, int]:
    user_indices, user_coords = run_tsne(user_vectors, max_points=tsne_user_sample, seed=seed, n_components=n_components)
    item_indices, item_coords = run_tsne(item_vectors, max_points=tsne_item_sample, seed=seed, n_components=n_components)
    output_suffix = build_output_suffix("tsne", n_components)

    sampled_user_metadata = user_metadata.iloc[user_indices].reset_index(drop=True)
    sampled_item_metadata = item_metadata.iloc[item_indices].reset_index(drop=True)

    save_embedding_csv(sampled_user_metadata, user_coords, str(RESULT_DIR / f"{model_name}_user_embedding_{output_suffix}.csv"), prefix=output_suffix)
    save_embedding_csv(sampled_item_metadata, item_coords, str(RESULT_DIR / f"{model_name}_item_embedding_{output_suffix}.csv"), prefix=output_suffix)

    plot_user_embeddings(sampled_user_metadata, user_coords, str(FIGURE_DIR / f"{model_name}_user_embedding_{output_suffix}.png"))
    plot_item_embeddings(sampled_item_metadata, item_coords, str(FIGURE_DIR / f"{model_name}_item_embedding_{output_suffix}.png"))
    plot_joint_embeddings(user_coords, item_coords, str(FIGURE_DIR / f"{model_name}_joint_embedding_{output_suffix}.png"))
    return len(user_indices), len(item_indices)


def run_embedding_visualization(
    model_name: str = "graph_augmented_twotower",
    checkpoint: str = "",
    method: str = "both",
    dimensions: int = 2,
    item_ablation: str = "all_features",
    device_name: str = "auto",
    seed: int = 42,
    tsne_user_sample: int = 1500,
    tsne_item_sample: int = 2500,
) -> None:
    if method not in {"pca", "tsne", "both"}:
        raise ValueError(f"Unsupported method: {method}")
    if dimensions not in {2, 3}:
        raise ValueError(f"Unsupported dimensions: {dimensions}")
    if item_ablation not in ITEM_ABLATION_MODES:
        raise ValueError(f"Unsupported item ablation mode: {item_ablation}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = select_device(device_name)
    checkpoint_path = resolve_checkpoint_path(model_name, checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    preprocessor, data, user_df, item_df = prepare_embedding_data(device=device, item_ablation=item_ablation)
    user_metadata, item_metadata = build_embedding_metadata(preprocessor, data, user_df, item_df)
    model = build_and_load_model(model_name=model_name, data=data, device=device, checkpoint_path=checkpoint_path)
    user_vectors, item_vectors = extract_model_embeddings(model=model, model_name=model_name, data=data, device=device)

    print(f"User embedding shape: {user_vectors.shape}")
    print(f"Item embedding shape: {item_vectors.shape}")

    if method in {"pca", "both"}:
        explained = export_pca_outputs(
            model_name=model_name,
            seed=seed,
            user_vectors=user_vectors,
            item_vectors=item_vectors,
            user_metadata=user_metadata,
            item_metadata=item_metadata,
            n_components=dimensions,
        )
        print(f"PCA finished. Explained variance ratio: {np.round(explained, 4).tolist()} (sum={explained.sum():.4f})")

    if method in {"tsne", "both"}:
        user_sample_count, item_sample_count = export_tsne_outputs(
            model_name=model_name,
            seed=seed,
            user_vectors=user_vectors,
            item_vectors=item_vectors,
            user_metadata=user_metadata,
            item_metadata=item_metadata,
            tsne_user_sample=tsne_user_sample,
            tsne_item_sample=tsne_item_sample,
            n_components=dimensions,
        )
        print(f"t-SNE finished. User samples: {user_sample_count}, Item samples: {item_sample_count}")

    print("Embedding visualization finished.")


def generate_paper_embedding_assets() -> None:
    run_embedding_visualization(model_name="graph_augmented_twotower", method="tsne", dimensions=3)
    save_current_figure_as_jpg("graph_augmented_twotower_joint_embedding_tsne_3d.png", FIGURE_SEQUENCE[11])
    save_current_figure_as_jpg("graph_augmented_twotower_user_embedding_tsne_3d.png", FIGURE_SEQUENCE[12])
    save_current_figure_as_jpg("graph_augmented_twotower_item_embedding_tsne_3d.png", FIGURE_SEQUENCE[13])


def update_markdown_image_links() -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    matches = list(re.finditer(r"!\[\]\((?:images|figures)/[^)]+\)", text))
    if len(matches) != len(FIGURE_SEQUENCE):
        raise ValueError(f"Expected {len(FIGURE_SEQUENCE)} image refs, found {len(matches)}.")

    replacements = [f"![](figures/{name})" for name in FIGURE_SEQUENCE]
    rebuilt: list[str] = []
    last_end = 0
    for match, replacement in zip(matches, replacements):
        rebuilt.append(text[last_end:match.start()])
        rebuilt.append(replacement)
        last_end = match.end()
    rebuilt.append(text[last_end:])
    MD_PATH.write_text("".join(rebuilt), encoding="utf-8")


def generate_paper_figures(update_markdown: bool = True) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    reviews_df, user_feat_df, item_df, item_feat_df = load_dataset_tables()
    item_popularity = reviews_df.groupby("asin").size()

    plot_yearly_review_trend(reviews_df, FIGURE_SEQUENCE[0])
    plot_rating_share(reviews_df, FIGURE_SEQUENCE[1])
    plot_review_text_length_histogram(reviews_df, FIGURE_SEQUENCE[2])
    plot_user_activity_distribution(user_feat_df, FIGURE_SEQUENCE[3])
    plot_user_avg_review_text_length_distribution(user_feat_df, FIGURE_SEQUENCE[4])
    plot_user_brand_preference_top15(user_feat_df, FIGURE_SEQUENCE[5])
    plot_item_metadata_feature_coverage(item_df, FIGURE_SEQUENCE[6])
    plot_item_image_count_distribution(item_df, FIGURE_SEQUENCE[7])
    plot_item_brand_top20_share(item_df, FIGURE_SEQUENCE[8])
    plot_item_vector_projection(item_feat_df, item_popularity, "image_vector", FIGURE_SEQUENCE[9])
    plot_item_vector_projection(item_feat_df, item_popularity, "title_vector", FIGURE_SEQUENCE[10])
    generate_paper_embedding_assets()

    if update_markdown:
        update_markdown_image_links()

    print("Generated figures:")
    for name in FIGURE_SEQUENCE:
        print(f" - figures/{name}")


def main() -> None:
    generate_paper_figures(update_markdown=True)


if __name__ == "__main__":
    main()
