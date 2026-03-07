import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import ast
import seaborn as sns
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print("Loading feature data...")
feat_df = pd.read_csv('new_feat/item.csv')
print(f"Loaded {len(feat_df)} items")

def parse_vector(vec_str):
    """解析向量字符串为numpy数组"""
    if isinstance(vec_str, str):
        return np.array(ast.literal_eval(vec_str))
    elif isinstance(vec_str, list):
        return np.array(vec_str)
    else:
        return np.zeros(512)

print("\nParsing vectors...")
title_vectors = np.array([parse_vector(v) for v in feat_df['title_clip']])
image_vectors = np.array([parse_vector(v) for v in feat_df['image_clip']])

print(f"Title vectors shape: {title_vectors.shape}")
print(f"Image vectors shape: {image_vectors.shape}")

print("\nApplying T-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)

title_tsne = tsne.fit_transform(title_vectors)
print(f"Title T-SNE shape: {title_tsne.shape}")

image_tsne = tsne.fit_transform(image_vectors)
print(f"Image T-SNE shape: {image_tsne.shape}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

scatter1 = axes[0].scatter(title_tsne[:, 0], title_tsne[:, 1], 
                          c=np.arange(len(title_tsne)), cmap='viridis', 
                          alpha=0.6, s=10)
axes[0].set_xlabel('T-SNE维度1', fontsize=12)
axes[0].set_ylabel('T-SNE维度2', fontsize=12)
axes[0].grid(True, alpha=0.3)

scatter2 = axes[1].scatter(image_tsne[:, 0], image_tsne[:, 1], 
                          c=np.arange(len(image_tsne)), cmap='plasma', 
                          alpha=0.6, s=10)
axes[1].set_xlabel('T-SNE维度1', fontsize=12)
axes[1].set_ylabel('T-SNE维度2', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization to tsne_visualization.png")

plt.figure(figsize=(12, 10))
plt.scatter(title_tsne[:, 0], title_tsne[:, 1], 
            c='blue', alpha=0.4, s=15, label='标题向量')
plt.scatter(image_tsne[:, 0], image_tsne[:, 1], 
            c='red', alpha=0.4, s=15, label='图像向量')
plt.xlabel('T-SNE维度1', fontsize=14)
plt.ylabel('T-SNE维度2', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_comparison.png', dpi=300, bbox_inches='tight')
print("Saved comparison to tsne_comparison.png")

sample_size = min(2000, len(feat_df))
sample_indices = np.random.choice(len(feat_df), sample_size, replace=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

title_sample_tsne = title_tsne[sample_indices]
image_sample_tsne = image_tsne[sample_indices]

scatter1 = axes[0].scatter(title_sample_tsne[:, 0], title_sample_tsne[:, 1], 
                          c=np.arange(sample_size), cmap='viridis', 
                          alpha=0.7, s=20)
axes[0].set_xlabel('T-SNE维度1', fontsize=12)
axes[0].set_ylabel('T-SNE维度2', fontsize=12)
axes[0].grid(True, alpha=0.3)

scatter2 = axes[1].scatter(image_sample_tsne[:, 0], image_sample_tsne[:, 1], 
                          c=np.arange(sample_size), cmap='plasma', 
                          alpha=0.7, s=20)
axes[1].set_xlabel('T-SNE维度1', fontsize=12)
axes[1].set_ylabel('T-SNE维度2', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tsne_sample.png', dpi=300, bbox_inches='tight')
print("Saved sample visualization to tsne_sample.png")

print("\n=== T-SNE Statistics ===")
print(f"Title vectors T-SNE range:")
print(f"  X: [{title_tsne[:, 0].min():.2f}, {title_tsne[:, 0].max():.2f}]")
print(f"  Y: [{title_tsne[:, 1].min():.2f}, {title_tsne[:, 1].max():.2f}]")

print(f"\nImage vectors T-SNE range:")
print(f"  X: [{image_tsne[:, 0].min():.2f}, {image_tsne[:, 0].max():.2f}]")
print(f"  Y: [{image_tsne[:, 1].min():.2f}, {image_tsne[:, 1].max():.2f}]")

print(f"\nVector similarity analysis:")
similarities = []
for i in range(min(100, len(title_vectors))):
    similarity = np.dot(title_vectors[i], image_vectors[i]) / (
        np.linalg.norm(title_vectors[i]) * np.linalg.norm(image_vectors[i])
    )
    similarities.append(similarity)

print(f"  Average cosine similarity (first 100 items): {np.mean(similarities):.4f}")
print(f"  Min similarity: {np.min(similarities):.4f}")
print(f"  Max similarity: {np.max(similarities):.4f}")

print("\n=== Visualization completed ===")
print("Generated files:")
print("  - tsne_visualization.png (side-by-side comparison)")
print("  - tsne_comparison.png (overlaid comparison)")
print("  - tsne_sample.png (sample visualization)")
