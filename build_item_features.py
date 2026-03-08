import pandas as pd
import numpy as np
import os
import re
import torch
import clip
from PIL import Image
from collections import Counter
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

os.makedirs('new_feat', exist_ok=True)

print("Loading data...")
items_df = pd.read_csv('new_dataset/item.csv')
print(f"Loaded {len(items_df)} items")

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"CLIP model loaded on {device}")

common_stop_words = {
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

def extract_keywords(title):
    if not isinstance(title, str) or not title:
        return ""
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
    words = [word for word in words if word not in common_stop_words]
    
    word_freq = Counter(words)
    top_keywords = [word for word, freq in word_freq.most_common(10)]
    
    return ' '.join(top_keywords)

def encode_text_with_clip(text):
    if not isinstance(text, str) or not text:
        return None
    
    text = text[:77]
    text_tokens = clip.tokenize([text], truncate=True).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()[0]

def load_and_encode_image(asin):
    if not isinstance(asin, str) or not asin:
        return None
    
    try:
        image_path = os.path.join('images', f'{asin}.jpg')
        
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
    except Exception as e:
        pass
    
    return None

def apply_tsne(vectors, n_components=3):
    if len(vectors) == 0:
        return np.array([])
    
    if len(vectors) < n_components:
        return np.array([np.zeros(n_components) for _ in range(len(vectors))])
    
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(vectors) - 1))
    reduced_vectors = tsne.fit_transform(vectors)
    
    return reduced_vectors

def vector_to_list(vector):
    if vector is None:
        return []
    return vector.tolist()

print("Processing items...")
results = []

title_clip_vectors = []
image_clip_vectors = []
feature_clip_vectors = []
description_clip_vectors = []

for idx, row in items_df.iterrows():
    if idx % 100 == 0:
        print(f"Processing {idx}/{len(items_df)}")
    
    asin = row['asin']
    
    result = {'asin': asin}
    
    for col in items_df.columns:
        if col == 'asin':
            continue
        
        value = row[col]
        
        if pd.isna(value):
            if items_df[col].dtype in ['float64', 'int64']:
                result[col] = items_df[col].mean()
            else:
                result[col] = ''
        else:
            result[col] = value
    
    title = result.get('title', '')
    result['title_keywords'] = extract_keywords(title)
    
    title_clip = encode_text_with_clip(title)
    title_clip_vectors.append(title_clip)
    
    image_clip = load_and_encode_image(asin)
    image_clip_vectors.append(image_clip)
    
    feature_text = result.get('feature', '')
    feature_clip = encode_text_with_clip(feature_text)
    feature_clip_vectors.append(feature_clip)
    
    description_text = result.get('description', '')
    description_clip = encode_text_with_clip(description_text)
    description_clip_vectors.append(description_clip)
    
    results.append(result)

print("Handling missing vectors...")
title_clip_vectors = np.array([v if v is not None else np.zeros(512) for v in title_clip_vectors])
image_clip_vectors = np.array([v if v is not None else np.zeros(512) for v in image_clip_vectors])
feature_clip_vectors = np.array([v if v is not None else np.zeros(512) for v in feature_clip_vectors])
description_clip_vectors = np.array([v if v is not None else np.zeros(512) for v in description_clip_vectors])

valid_image_indices = [i for i, v in enumerate(image_clip_vectors) if not np.all(v == 0)]
if len(valid_image_indices) > 0:
    mean_image_vector = np.mean(image_clip_vectors[valid_image_indices], axis=0)
    for i, v in enumerate(image_clip_vectors):
        if np.all(v == 0):
            image_clip_vectors[i] = mean_image_vector

print("Applying TSNE dimensionality reduction...")
print("Reducing title vectors to 3D...")
title_tsne_vectors = apply_tsne(title_clip_vectors, n_components=3)

print("Reducing image vectors to 3D...")
image_tsne_vectors = apply_tsne(image_clip_vectors, n_components=3)

print("Reducing feature vectors to 3D...")
feature_tsne_vectors = apply_tsne(feature_clip_vectors, n_components=3)

print("Reducing description vectors to 3D...")
description_tsne_vectors = apply_tsne(description_clip_vectors, n_components=3)

print("Creating DataFrame...")
feat_df = pd.DataFrame(results)

print("Adding reduced vectors to DataFrame...")
feat_df['title_vector'] = [vector_to_list(v) for v in title_tsne_vectors]
feat_df['image_vector'] = [vector_to_list(v) for v in image_tsne_vectors]
feat_df['feature_vector'] = [vector_to_list(v) for v in feature_tsne_vectors]
feat_df['description_vector'] = [vector_to_list(v) for v in description_tsne_vectors]

print("Filling remaining missing values...")
for col in feat_df.columns:
    if col in ['asin', 'title_vector', 'image_vector', 'feature_vector', 'description_vector']:
        continue
    
    if feat_df[col].dtype in ['float64', 'int64']:
        feat_df[col].fillna(feat_df[col].mean(), inplace=True)
    else:
        feat_df[col].fillna('', inplace=True)

print("Saving item features to CSV...")
feat_df.to_csv('new_feat/item.csv', index=False, encoding='utf-8')
print(f"Saved {len(feat_df)} items to new_feat/item.csv")

print("\n=== Feature Statistics ===")
print(f"Total items: {len(feat_df)}")
print(f"Total features: {len(feat_df.columns)}")

print(f"\nTitle TSNE vectors:")
print(f"  Shape: {title_tsne_vectors.shape}")
print(f"  Mean: {title_tsne_vectors.mean():.6f}")
print(f"  Std: {title_tsne_vectors.std():.6f}")

print(f"\nImage TSNE vectors:")
print(f"  Shape: {image_tsne_vectors.shape}")
print(f"  Mean: {image_tsne_vectors.mean():.6f}")
print(f"  Std: {image_tsne_vectors.std():.6f}")

print(f"\nFeature TSNE vectors:")
print(f"  Shape: {feature_tsne_vectors.shape}")
print(f"  Mean: {feature_tsne_vectors.mean():.6f}")
print(f"  Std: {feature_tsne_vectors.std():.6f}")

print(f"\nDescription TSNE vectors:")
print(f"  Shape: {description_tsne_vectors.shape}")
print(f"  Mean: {description_tsne_vectors.mean():.6f}")
print(f"  Std: {description_tsne_vectors.std():.6f}")

print(f"\n=== Feature columns ===")
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
print("  - new_feat/item.csv (item features with 3D vectors stored as lists)")
