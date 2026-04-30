"""构建商品侧多模态特征。

输出结构与当前训练流程完全对齐，包括：
1. 商品元数据中的数值特征；
2. 标题关键词；
3. 标题、图像、特征、描述四种模态的 64 维向量。
"""

import os
import re
from collections import Counter

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA

OUTPUT_DIR = 'new_feat'
ITEM_OUTPUT_PATH = f'{OUTPUT_DIR}/item.csv'
VECTOR_DIM = 512
REDUCED_DIM = 64
TEXT_BATCH_SIZE = 64
IMAGE_BATCH_SIZE = 32

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


def extract_keywords(text, top_k=10):
    """从商品标题中提取紧凑的关键词摘要。"""
    if not isinstance(text, str) or not text:
        return ""

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [word for word in words if word not in COMMON_STOP_WORDS]
    top_keywords = [word for word, _ in Counter(filtered_words).most_common(top_k)]
    return ' '.join(top_keywords)


def apply_pca(vectors, n_components=REDUCED_DIM):
    """将 CLIP 向量降到当前项目使用的 PCA 紧凑表示。"""
    if len(vectors) == 0:
        return np.array([])

    if len(vectors) < n_components:
        return np.zeros((len(vectors), n_components), dtype=np.float32)

    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(vectors).astype(np.float32)


def vector_to_list(vector):
    """将稠密向量转换为适合写入 CSV 的列表格式。"""
    if vector is None:
        return []
    return vector.tolist()


def count_delimited_values(value, delimiter='|'):
    """统计以分隔符拼接的列表字段长度，空值返回 0。"""
    if not isinstance(value, str):
        return 0

    tokens = [token.strip() for token in value.split(delimiter) if token.strip()]
    return len(tokens)


def measure_text_length(text):
    """计算文本字段长度，用于构造额外的结构化统计特征。"""
    if not isinstance(text, str):
        return 0
    return len(text.strip())


def fill_remaining_missing_values(df, skip_columns):
    """在多模态特征构建完成后补齐非向量字段缺失值。"""
    for col in df.columns:
        if col in skip_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna('')
    return df


def prepare_item_records(items_df):
    """在多模态编码前逐行整理商品元数据。"""
    records = []
    numeric_means = items_df.select_dtypes(include=[np.number]).mean().to_dict()

    for idx, row in items_df.iterrows():
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(items_df)}")

        record = {'asin': row['asin']}
        for col in items_df.columns:
            if col == 'asin':
                continue
            value = row[col]
            if pd.isna(value):
                record[col] = numeric_means.get(col, '') if col in numeric_means else ''
            else:
                record[col] = value
        record['title_keywords'] = extract_keywords(record.get('title', ''))
        records.append(record)

    return records


def batch_encode_texts(model, device, texts, batch_size=TEXT_BATCH_SIZE):
    """以批处理方式编码商品文本字段。"""
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

    return features


def batch_encode_images(model, preprocess, device, asins, batch_size=IMAGE_BATCH_SIZE):
    """批量编码本地图像，并返回哪些商品真正拥有可用图像。"""
    features = np.zeros((len(asins), VECTOR_DIM), dtype=np.float32)
    image_tensors = []
    image_indices = []

    for idx, asin in enumerate(asins):
        if not isinstance(asin, str) or not asin:
            continue

        image_path = os.path.join('images', f'{asin}.jpg')
        if not os.path.exists(image_path):
            continue

        try:
            with Image.open(image_path) as image:
                image_tensor = preprocess(image.convert('RGB'))
            image_tensors.append(image_tensor)
            image_indices.append(idx)
        except Exception:
            continue

    for start in range(0, len(image_tensors), batch_size):
        batch_tensors = torch.stack(image_tensors[start:start + batch_size]).to(device)
        batch_indices = image_indices[start:start + batch_size]

        with torch.no_grad():
            image_features = model.encode_image(batch_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        features[batch_indices] = image_features.cpu().numpy()

    valid_mask = np.zeros(len(asins), dtype=np.float32)
    if image_indices:
        valid_mask[image_indices] = 1.0

    return features, valid_mask


def main():
    """生成包含文本与图像信息的 `new_feat/item.csv`。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    items_df = pd.read_csv('new_dataset/item.csv')
    items_df = items_df.drop_duplicates(subset=['asin']).reset_index(drop=True)
    print(f"Loaded {len(items_df)} items")

    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print(f"CLIP model loaded on {device}")

    # 第一阶段：整理原始商品字段，保证输出模式稳定。
    print("Processing items...")
    records = prepare_item_records(items_df)
    feat_df = pd.DataFrame(records)

    title_texts = feat_df['title'].fillna('').astype(str).tolist()
    feature_texts = feat_df['feature'].fillna('').astype(str).tolist()
    description_texts = feat_df['description'].fillna('').astype(str).tolist()
    asins = feat_df['asin'].astype(str).tolist()

    # 第二阶段：批量编码三种文本模态和本地图像模态。
    print("Encoding text fields with CLIP...")
    title_clip_vectors = batch_encode_texts(model, device, title_texts)
    feature_clip_vectors = batch_encode_texts(model, device, feature_texts)
    description_clip_vectors = batch_encode_texts(model, device, description_texts)

    print("Encoding item images with CLIP...")
    image_clip_vectors, image_valid_mask = batch_encode_images(model, preprocess, device, asins)

    # 第三阶段：将高维 CLIP 向量压缩为模型当前使用的 64 维表示。
    print("Applying PCA dimensionality reduction...")
    print(f"Reducing title vectors to {REDUCED_DIM}D...")
    title_pca_vectors = apply_pca(title_clip_vectors, n_components=REDUCED_DIM)

    print(f"Reducing image vectors to {REDUCED_DIM}D...")
    image_pca_vectors = apply_pca(image_clip_vectors, n_components=REDUCED_DIM)

    print(f"Reducing feature vectors to {REDUCED_DIM}D...")
    feature_pca_vectors = apply_pca(feature_clip_vectors, n_components=REDUCED_DIM)

    print(f"Reducing description vectors to {REDUCED_DIM}D...")
    description_pca_vectors = apply_pca(description_clip_vectors, n_components=REDUCED_DIM)

    print("Adding reduced vectors to DataFrame...")
    feat_df['title_vector'] = [vector_to_list(v) for v in title_pca_vectors]
    feat_df['image_vector'] = [vector_to_list(v) for v in image_pca_vectors]
    feat_df['feature_vector'] = [vector_to_list(v) for v in feature_pca_vectors]
    feat_df['description_vector'] = [vector_to_list(v) for v in description_pca_vectors]
    feat_df['has_image'] = image_valid_mask.astype(np.float32)
    feat_df['has_price'] = feat_df['price'].fillna('').astype(str).str.strip().ne('').astype(np.float32)
    feat_df['has_feature'] = feat_df['feature'].fillna('').astype(str).str.strip().ne('').astype(np.float32)
    feat_df['has_description'] = feat_df['description'].fillna('').astype(str).str.strip().ne('').astype(np.float32)
    feat_df['title_length'] = feat_df['title'].apply(measure_text_length).astype(np.float32)
    feat_df['feature_length'] = feat_df['feature'].apply(measure_text_length).astype(np.float32)
    feat_df['description_length'] = feat_df['description'].apply(measure_text_length).astype(np.float32)
    feat_df['also_view_count'] = feat_df['also_view'].apply(count_delimited_values).astype(np.float32)
    feat_df['also_buy_count'] = feat_df['also_buy'].apply(count_delimited_values).astype(np.float32)

    print("Filling remaining missing values...")
    feat_df = fill_remaining_missing_values(
        feat_df,
        skip_columns={'asin', 'title_vector', 'image_vector', 'feature_vector', 'description_vector'}
    )

    # 第四阶段：输出 DataPreprocessor 直接使用的商品特征表。
    print("Saving item features to CSV...")
    feat_df.to_csv(ITEM_OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"Saved {len(feat_df)} items to {ITEM_OUTPUT_PATH}")

    print("\n=== Feature Statistics ===")
    print(f"Total items: {len(feat_df)}")
    print(f"Total features: {len(feat_df.columns)}")

    print("\nTitle PCA vectors:")
    print(f"  Shape: {title_pca_vectors.shape}")
    print(f"  Mean: {title_pca_vectors.mean():.6f}")
    print(f"  Std: {title_pca_vectors.std():.6f}")

    print("\nImage PCA vectors:")
    print(f"  Shape: {image_pca_vectors.shape}")
    print(f"  Mean: {image_pca_vectors.mean():.6f}")
    print(f"  Std: {image_pca_vectors.std():.6f}")

    print("\nFeature PCA vectors:")
    print(f"  Shape: {feature_pca_vectors.shape}")
    print(f"  Mean: {feature_pca_vectors.mean():.6f}")
    print(f"  Std: {feature_pca_vectors.std():.6f}")

    print("\nDescription PCA vectors:")
    print(f"  Shape: {description_pca_vectors.shape}")
    print(f"  Mean: {description_pca_vectors.mean():.6f}")
    print(f"  Std: {description_pca_vectors.std():.6f}")

    print("\n=== Feature columns ===")
    print(f"Columns in item.csv: {list(feat_df.columns)}")

    print("\n=== Sample data ===")
    print(f"First item ASIN: {feat_df.iloc[0]['asin']}")
    print(f"Title keywords: {feat_df.iloc[0]['title_keywords']}")
    print(f"Title vector (first 3 dims): {feat_df.iloc[0]['title_vector']}")
    print(f"Image vector (first 3 dims): {feat_df.iloc[0]['image_vector']}")
    print(f"Feature vector (first 3 dims): {feat_df.iloc[0]['feature_vector']}")
    print(f"Description vector (first 3 dims): {feat_df.iloc[0]['description_vector']}")

    print("\n=== Completed ===")
    print("Generated files:")
    print(f"  - new_feat/item.csv (item features with {REDUCED_DIM}D PCA vectors stored as lists)")


if __name__ == '__main__':
    main()
