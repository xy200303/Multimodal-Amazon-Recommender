import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print("Loading user features...")
user_feat_df = pd.read_csv('new_feat/user.csv')
print(f"Loaded {len(user_feat_df)} users")

print("Loading content vectors...")
content_vectors = np.load('new_feat/user_content_feat.npy')
print(f"Content vectors shape: {content_vectors.shape}")

print("\nApplying T-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
user_tsne = tsne.fit_transform(content_vectors)
print(f"User T-SNE shape: {user_tsne.shape}")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

scatter1 = axes[0, 0].scatter(user_tsne[:, 0], user_tsne[:, 1], 
                          c=user_feat_df['avg_rating'], cmap='viridis', 
                          alpha=0.6, s=15)
axes[0, 0].set_xlabel('T-SNE维度1', fontsize=12)
axes[0, 0].set_ylabel('T-SNE维度2', fontsize=12)
axes[0, 0].set_title('用户分布（按评分）', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
cbar1.set_label('评分', fontsize=11)

scatter2 = axes[0, 1].scatter(user_tsne[:, 0], user_tsne[:, 1], 
                          c=user_feat_df['review_count'], cmap='plasma', 
                          alpha=0.6, s=15)
axes[0, 1].set_xlabel('T-SNE维度1', fontsize=12)
axes[0, 1].set_ylabel('T-SNE维度2', fontsize=12)
axes[0, 1].set_title('用户分布（按评论数）', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
cbar2.set_label('评论数', fontsize=11)

scatter3 = axes[1, 0].scatter(user_tsne[:, 0], user_tsne[:, 1], 
                          c=user_feat_df['avg_text_length'], cmap='coolwarm', 
                          alpha=0.6, s=15)
axes[1, 0].set_xlabel('T-SNE维度1', fontsize=12)
axes[1, 0].set_ylabel('T-SNE维度2', fontsize=12)
axes[1, 0].set_title('用户分布（按文本长度）', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
cbar3 = plt.colorbar(scatter3, ax=axes[1, 0])
cbar3.set_label('平均文本长度', fontsize=11)

scatter4 = axes[1, 1].scatter(user_tsne[:, 0], user_tsne[:, 1], 
                          c=user_feat_df['verified_ratio'], cmap='RdYlBu', 
                          alpha=0.6, s=15)
axes[1, 1].set_xlabel('T-SNE维度1', fontsize=12)
axes[1, 1].set_ylabel('T-SNE维度2', fontsize=12)
axes[1, 1].set_title('用户分布（按验证比例）', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
cbar4 = plt.colorbar(scatter4, ax=axes[1, 1])
cbar4.set_label('验证比例', fontsize=11)

plt.tight_layout()
plt.savefig('user_tsne_visualization.png', dpi=300, bbox_inches='tight')
print("Saved visualization to user_tsne_visualization.png")

plt.figure(figsize=(14, 12))
plt.scatter(user_tsne[:, 0], user_tsne[:, 1], 
            c=user_feat_df['avg_rating'], cmap='viridis', 
            alpha=0.5, s=20)
plt.xlabel('T-SNE维度1', fontsize=14)
plt.ylabel('T-SNE维度2', fontsize=14)
plt.colorbar(label='评分')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('user_tsne_single.png', dpi=300, bbox_inches='tight')
print("Saved single visualization to user_tsne_single.png")

print("\n=== T-SNE Statistics ===")
print(f"User vectors T-SNE range:")
print(f"  X: [{user_tsne[:, 0].min():.2f}, {user_tsne[:, 0].max():.2f}]")
print(f"  Y: [{user_tsne[:, 1].min():.2f}, {user_tsne[:, 1].max():.2f}]")

print(f"\nUser feature statistics:")
print(f"  Average rating: {user_feat_df['avg_rating'].mean():.2f}")
print(f"  Rating std: {user_feat_df['rating_std'].mean():.2f}")
print(f"  Average review count: {user_feat_df['review_count'].mean():.1f}")
print(f"  Average text length: {user_feat_df['avg_text_length'].mean():.1f}")
print(f"  Average verified ratio: {user_feat_df['verified_ratio'].mean():.3f}")

print(f"\nRating distribution:")
rating_counts = user_feat_df['avg_rating'].value_counts().sort_index()
print(f"  Rating 1.0: {rating_counts.get(1.0, 0)} users")
print(f"  Rating 2.0: {rating_counts.get(2.0, 0)} users")
print(f"  Rating 3.0: {rating_counts.get(3.0, 0)} users")
print(f"  Rating 4.0: {rating_counts.get(4.0, 0)} users")
print(f"  Rating 5.0: {rating_counts.get(5.0, 0)} users")

print("\n=== Visualization completed ===")
print("Generated files:")
print("  - user_tsne_visualization.png (4-panel comparison)")
print("  - user_tsne_single.png (single scatter plot)")
