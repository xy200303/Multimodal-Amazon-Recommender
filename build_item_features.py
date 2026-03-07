import pandas as pd
import numpy as np
import os
import re
import torch
import clip
from PIL import Image
from collections import Counter
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
    """从标题中提取关键词"""
    if not isinstance(title, str) or not title:
        return ""
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
    words = [word for word in words if word not in common_stop_words]
    
    word_freq = Counter(words)
    top_keywords = [word for word, freq in word_freq.most_common(10)]
    
    return ' '.join(top_keywords)

def encode_text_with_clip(text):
    """使用CLIP对文本进行编码"""
    if not isinstance(text, str) or not text:
        return None
    
    text = text[:77]
    text_tokens = clip.tokenize([text], truncate=True).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()[0]

def load_and_encode_image(asin):
    """加载图像并使用CLIP编码"""
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

print("Processing items...")
results = []
title_vectors_list = []
image_vectors_list = []
title_indices = []
image_indices = []

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
    
    text_vector = encode_text_with_clip(title)
    if text_vector is not None:
        title_vectors_list.append(text_vector)
        title_indices.append(idx)
        result['title_feat_idx'] = idx
    else:
        result['title_feat_idx'] = -1
    
    image_vector = load_and_encode_image(asin)
    if image_vector is not None:
        image_vectors_list.append(image_vector)
        image_indices.append(idx)
        result['image_feat_idx'] = idx
    else:
        result['image_feat_idx'] = -1
    
    results.append(result)

print("Creating DataFrame...")
feat_df = pd.DataFrame(results)

print("Handling missing image vectors...")
if len(image_vectors_list) > 0:
    mean_image_vector = np.mean(image_vectors_list, axis=0)
    for idx, row in feat_df.iterrows():
        if row['image_feat_idx'] == -1:
            image_vectors_list.append(mean_image_vector)
            image_indices.append(idx)
            feat_df.at[idx, 'image_feat_idx'] = idx
else:
    for idx, row in feat_df.iterrows():
        if row['image_feat_idx'] == -1:
            image_vectors_list.append(np.zeros(512))
            image_indices.append(idx)
            feat_df.at[idx, 'image_feat_idx'] = idx

print("Handling missing title vectors...")
if len(title_vectors_list) > 0:
    mean_title_vector = np.mean(title_vectors_list, axis=0)
    for idx, row in feat_df.iterrows():
        if row['title_feat_idx'] == -1:
            title_vectors_list.append(mean_title_vector)
            title_indices.append(idx)
            feat_df.at[idx, 'title_feat_idx'] = idx
else:
    for idx, row in feat_df.iterrows():
        if row['title_feat_idx'] == -1:
            title_vectors_list.append(np.zeros(512))
            title_indices.append(idx)
            feat_df.at[idx, 'title_feat_idx'] = idx

print("Filling remaining missing values...")
for col in feat_df.columns:
    if col in ['asin', 'title_feat_idx', 'image_feat_idx']:
        continue
    
    if feat_df[col].dtype in ['float64', 'int64']:
        feat_df[col].fillna(feat_df[col].mean(), inplace=True)
    else:
        feat_df[col].fillna('', inplace=True)

print("Saving feature vectors to numpy files...")
title_vectors_array = np.array(title_vectors_list)
image_vectors_array = np.array(image_vectors_list)

print(f"Title vectors shape: {title_vectors_array.shape}")
print(f"Image vectors shape: {image_vectors_array.shape}")

np.save('new_feat/item_title_feat.npy', title_vectors_array)
print("Saved title vectors to new_feat/item_title_feat.npy")

np.save('new_feat/item_image_feat.npy', image_vectors_array)
print("Saved image vectors to new_feat/item_image_feat.npy")

print("Saving item features to CSV...")
feat_df.to_csv('new_feat/item.csv', index=False, encoding='utf-8')
print(f"Saved {len(feat_df)} items to new_feat/item.csv")

print("\n=== Feature Statistics ===")
print(f"Total items: {len(feat_df)}")
print(f"Total features: {len(feat_df.columns)}")

print(f"\nTitle vectors:")
print(f"  Shape: {title_vectors_array.shape}")
print(f"  Mean: {title_vectors_array.mean():.6f}")
print(f"  Std: {title_vectors_array.std():.6f}")
print(f"  Min: {title_vectors_array.min():.6f}")
print(f"  Max: {title_vectors_array.max():.6f}")

print(f"\nImage vectors:")
print(f"  Shape: {image_vectors_array.shape}")
print(f"  Mean: {image_vectors_array.mean():.6f}")
print(f"  Std: {image_vectors_array.std():.6f}")
print(f"  Min: {image_vectors_array.min():.6f}")
print(f"  Max: {image_vectors_array.max():.6f}")

print(f"\nIndex statistics:")
print(f"  Items with title vectors: {len([idx for idx in feat_df['title_feat_idx'] if idx >= 0])}")
print(f"  Items with image vectors: {len([idx for idx in feat_df['image_feat_idx'] if idx >= 0])}")
print(f"  Items missing title vectors: {len([idx for idx in feat_df['title_feat_idx'] if idx < 0])}")
print(f"  Items missing image vectors: {len([idx for idx in feat_df['image_feat_idx'] if idx < 0])}")

print("\n=== Feature columns ===")
print(f"Columns in item.csv: {list(feat_df.columns)}")

print("\n=== Completed ===")
print("Generated files:")
print("  - new_feat/item.csv (item features with indices)")
print("  - new_feat/item_title_feat.npy (title CLIP vectors)")
print("  - new_feat/item_image_feat.npy (image CLIP vectors)")
