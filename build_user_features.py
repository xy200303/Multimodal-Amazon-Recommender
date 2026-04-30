"""构建用户侧特征。

输出格式与当前项目训练流程保持一致，包括：
1. 手工统计的数值型特征；
2. 用户偏好的颜色/尺码词；
3. 通过 CLIP 编码并降到 64 维的文本向量。
"""

import os
import re
from collections import Counter

import clip
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

OUTPUT_DIR = 'new_feat'
USER_OUTPUT_PATH = f'{OUTPUT_DIR}/user.csv'
VECTOR_DIM = 512
REDUCED_DIM = 64
TEXT_BATCH_SIZE = 32

COMMON_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
    'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all',
    'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'also', 'now', 'here', 'there', 'then', 'once', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under',
    'again', 'further', 'while', 'up', 'down', 'off', 'over', 'out'
}

COLOR_KEYWORDS = [
    'black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple',
    'orange', 'brown', 'gray', 'silver', 'gold', 'beige', 'cream',
    'navy', 'tan', 'khaki', 'maroon', 'olive', 'teal', 'charcoal'
]

SIZE_KEYWORDS = [
    'small', 'medium', 'large', 'x-small', 'x-large', 'xx-small', 'xx-large',
    'one size', '2xl', '3xl', '4xl', '5xl', 'plus size', 'xs', 's', 'm', 'l', 'xl'
]


def build_keyword_pattern(keyword):
    """为颜色/尺码词构造边界更严格的匹配模式，避免 s/m/l 命中普通单词。"""
    escaped_keyword = re.escape(keyword).replace(r'\ ', r'\s+')
    return re.compile(rf'(?<![a-z0-9-]){escaped_keyword}(?![a-z0-9-])', re.IGNORECASE)


def extract_keywords(text, top_k=15):
    """从用户评论语料中提取高频关键词。"""
    if not isinstance(text, str) or not text:
        return ""

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [word for word in words if word not in COMMON_STOP_WORDS]
    top_keywords = [word for word, _ in Counter(filtered_words).most_common(top_k)]
    return ' '.join(top_keywords)


def extract_ranked_preferences(texts, keywords):
    """按出现频次抽取用户偏好的颜色/尺码词，并按频次从高到低排序。"""
    if not texts:
        return []

    keyword_counter = Counter()
    compiled_patterns = {keyword: build_keyword_pattern(keyword) for keyword in keywords}

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue

        normalized_text = re.sub(r'\s+', ' ', text.lower()).strip()
        for keyword, pattern in compiled_patterns.items():
            match_count = len(pattern.findall(normalized_text))
            if match_count > 0:
                keyword_counter[keyword] += match_count

    return [
        keyword for keyword, _ in sorted(
            keyword_counter.items(),
            key=lambda item: (-item[1], item[0])
        )
    ]


def apply_pca(vectors, n_components=REDUCED_DIM):
    """将高维 CLIP 向量压缩为当前项目使用的 PCA 低维表示。"""
    if len(vectors) == 0:
        return np.array([])

    if len(vectors) < n_components:
        return np.zeros((len(vectors), n_components), dtype=np.float32)

    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(vectors).astype(np.float32)


def vector_to_list(vector):
    """将向量转换为适合写入 CSV 的列表格式。"""
    if vector is None:
        return []
    return vector.tolist()


def fill_missing_values(df):
    """在用户特征聚合后补齐缺失值。"""
    for col in df.columns:
        if col == 'reviewerID':
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna('')
    return df


def batch_encode_texts(model, device, texts, batch_size=TEXT_BATCH_SIZE):
    """以批处理方式对用户评论文本做 CLIP 编码。"""
    features = np.zeros((len(texts), VECTOR_DIM), dtype=np.float32)
    valid_indices = [idx for idx, text in enumerate(texts) if isinstance(text, str) and text.strip()]


    for start in range(0, len(valid_indices), batch_size):
        batch_indices = valid_indices[start:start + batch_size]
        batch_texts = [texts[idx] for idx in batch_indices]
        text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        features[batch_indices] = text_features.cpu().numpy()

        processed = min(start + batch_size, len(valid_indices))
        if processed % 100 == 0 or processed == len(valid_indices):
            print(f"Processed {processed}/{len(valid_indices)} text entries")

    return features


def main():
    """聚合用户画像特征并输出 `new_feat/user.csv`。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading training reviews data...")
    reviews_df = pd.read_csv('new_dataset/train_reviews.csv')
    print(f"Loaded {len(reviews_df)} reviews")

    print("Loading item data...")
    items_df = pd.read_csv('new_dataset/item.csv')
    print(f"Loaded {len(items_df)} items")

    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    print(f"CLIP model loaded on {device}")

    # 预先构造快速查找表，避免在 groupby 过程中反复扫描整个商品表。
    item_brand_map = items_df.set_index('asin')['brand'].fillna('').to_dict()
    item_title_map = items_df.set_index('asin')['title'].fillna('').to_dict()

    # 在一次 groupby 遍历里同时完成统计特征构建和文本语料整理。
    print("Processing user features...")
    user_rows = []
    user_texts = []

    for user_id, group in reviews_df.groupby('reviewerID', sort=False):
        reviews = group.sort_values('unixReviewTime').reset_index(drop=True)
        review_texts = reviews['reviewText'].fillna('').astype(str)
        review_lengths = review_texts.str.len()
        user_asins = reviews['asin'].dropna().astype(str).unique().tolist()

        user_brands = [item_brand_map.get(asin, '') for asin in user_asins if item_brand_map.get(asin, '')]
        user_titles = [item_title_map.get(asin, '') for asin in user_asins if item_title_map.get(asin, '')]
        all_reviews_text = ' '.join(review_texts.tolist())
        ranked_colors = extract_ranked_preferences(user_titles, COLOR_KEYWORDS)
        ranked_sizes = extract_ranked_preferences(user_titles, SIZE_KEYWORDS)

        brand_counts = pd.Series(user_brands).value_counts() if user_brands else pd.Series(dtype='int64')

        # 这些统计量用于描述用户历史行为强度和评分风格。
        user_rows.append({
            'reviewerID': user_id,
            'reviewerName': reviews['reviewerName'].iloc[0] if len(reviews) > 0 else '',
            'review_count': len(reviews),
            'avg_rating': reviews['overall'].mean(),
            'rating_std': reviews['overall'].std(),
            'min_rating': reviews['overall'].min(),
            'max_rating': reviews['overall'].max(),
            'avg_text_length': review_lengths.mean(),
            'text_length_std': review_lengths.std(),
            'min_text_length': review_lengths.min(),
            'max_text_length': review_lengths.max(),
            'verified_count': reviews['verified'].sum(),
            'verified_ratio': reviews['verified'].mean(),
            'all_reviews_keywords': extract_keywords(all_reviews_text),
            'top_category': brand_counts.index[0] if not brand_counts.empty else '',
            'top_category_count': int(brand_counts.iloc[0]) if not brand_counts.empty else 0,
            'top_style_colors': ranked_colors,
            'top_style_sizes': ranked_sizes,
            'top_style_color_count': len(ranked_colors),
            'top_style_size_count': len(ranked_sizes),
        })
        user_texts.append(all_reviews_text)

    print("Creating DataFrame...")
    user_feat_df = pd.DataFrame(user_rows)

    print("Handling missing values...")
    user_feat_df = fill_missing_values(user_feat_df)

    # 对拼接后的用户评论文本做编码，并降到下游模型使用的 64 维向量。
    print("Processing CLIP encoding for user reviews...")
    print(f"Encoding {len(user_texts)} user texts...")
    content_clip_vectors = batch_encode_texts(model, device, user_texts)
    print(f"Content CLIP vectors shape: {content_clip_vectors.shape}")

    print("Applying PCA dimensionality reduction...")
    print(f"Reducing user content vectors to {REDUCED_DIM}D...")
    content_pca_vectors = apply_pca(content_clip_vectors, n_components=REDUCED_DIM)
    print(f"Content PCA vectors shape: {content_pca_vectors.shape}")

    print("Adding reduced vectors to DataFrame...")
    user_feat_df['content_vector'] = [vector_to_list(v) for v in content_pca_vectors]

    # 按 DataPreprocessor 所需的固定字段结构保存。
    print("Saving user features to CSV...")
    user_feat_df.to_csv(USER_OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"Saved {len(user_feat_df)} users to {USER_OUTPUT_PATH}")

    print("\n=== User Feature Statistics ===")
    print(f"Total users: {len(user_feat_df)}")
    print(f"Total features: {len(user_feat_df.columns)}")

    print("\nRating statistics:")
    print(f"  Average rating: {user_feat_df['avg_rating'].mean():.2f}")
    print(f"  Rating std: {user_feat_df['rating_std'].mean():.2f}")
    print(f"  Min rating: {user_feat_df['min_rating'].mean():.2f}")
    print(f"  Max rating: {user_feat_df['max_rating'].mean():.2f}")

    print("\nReview statistics:")
    print(f"  Average review count: {user_feat_df['review_count'].mean():.1f}")
    print(f"  Min review count: {user_feat_df['review_count'].min()}")
    print(f"  Max review count: {user_feat_df['review_count'].max()}")

    print("\nText statistics:")
    print(f"  Average text length: {user_feat_df['avg_text_length'].mean():.1f}")
    print(f"  Text length std: {user_feat_df['text_length_std'].mean():.1f}")

    print("\nItem preference statistics:")
    print(f"  Users with top_category: {(user_feat_df['top_category'] != '').sum()}")
    print(f"  Users with top_style_colors: {sum(len(colors) > 0 for colors in user_feat_df['top_style_colors'])}")
    print(f"  Users with top_style_sizes: {sum(len(sizes) > 0 for sizes in user_feat_df['top_style_sizes'])}")

    print("\nContent PCA vector statistics:")
    print(f"  Shape: {content_pca_vectors.shape}")
    print(f"  Mean: {content_pca_vectors.mean():.6f}")
    print(f"  Std: {content_pca_vectors.std():.6f}")
    print(f"  Min: {content_pca_vectors.min():.6f}")
    print(f"  Max: {content_pca_vectors.max():.6f}")

    print("\n=== Feature columns ===")
    print(f"Columns in user.csv: {list(user_feat_df.columns)}")

    print("\n=== Sample data ===")
    print(f"First user ID: {user_feat_df.iloc[0]['reviewerID']}")
    print(f"Content vector (first 3 dims): {user_feat_df.iloc[0]['content_vector']}")
    print(f"Review count: {user_feat_df.iloc[0]['review_count']}")
    print(f"Average rating: {user_feat_df.iloc[0]['avg_rating']:.2f}")

    print("\n=== Completed ===")
    print("Generated files:")
    print(f"  - new_feat/user.csv (user features with {REDUCED_DIM}D PCA content vectors stored as lists)")


if __name__ == '__main__':
    main()
