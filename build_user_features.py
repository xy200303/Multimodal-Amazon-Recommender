import pandas as pd
import numpy as np
import os
import re
import torch
import clip
from collections import Counter
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

os.makedirs('new_feat', exist_ok=True)

print("Loading training reviews data...")
reviews_df = pd.read_csv('new_dataset/train_reviews.csv')
print(f"Loaded {len(reviews_df)} reviews")

print("Loading item data...")
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

def extract_keywords(text):
    if not isinstance(text, str) or not text:
        return ""
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    words = [word for word in words if word not in common_stop_words]
    
    word_freq = Counter(words)
    top_keywords = [word for word, freq in word_freq.most_common(15)]
    
    return ' '.join(top_keywords)

def extract_color_from_text(text):
    if not isinstance(text, str) or not text:
        return []
    
    color_keywords = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 
                    'orange', 'brown', 'gray', 'silver', 'gold', 'beige', 'cream', 
                    'navy', 'tan', 'khaki', 'maroon', 'olive', 'teal', 'charcoal']
    
    text_lower = text.lower()
    colors_found = [color for color in color_keywords if color in text_lower]
    
    return colors_found

def extract_size_from_text(text):
    if not isinstance(text, str) or not text:
        return []
    
    size_keywords = ['small', 'medium', 'large', 'x-small', 'x-large', 'xx-small', 'xx-large',
                    'one size', '2xl', '3xl', '4xl', '5xl', 'plus size', 'xs', 's', 'm', 'l', 'xl']
    
    text_lower = text.lower()
    sizes_found = [size for size in size_keywords if size in text_lower]
    
    return sizes_found

def encode_text_with_clip(text):
    if not isinstance(text, str) or not text:
        return None
    
    text = text[:77]
    text_tokens = clip.tokenize([text], truncate=True).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()[0]

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

print("Processing user features...")
user_features = {}

for user_id, group in reviews_df.groupby('reviewerID'):
    reviews = group.sort_values('unixReviewTime')
    
    user_features[user_id] = {
        'reviewerID': user_id,
        'reviewerName': reviews['reviewerName'].iloc[0] if len(reviews) > 0 else '',
        'review_count': len(reviews),
        'avg_rating': reviews['overall'].mean(),
        'rating_std': reviews['overall'].std(),
        'min_rating': reviews['overall'].min(),
        'max_rating': reviews['overall'].max(),
        'avg_text_length': reviews['reviewText'].str.len().mean(),
        'text_length_std': reviews['reviewText'].str.len().std(),
        'min_text_length': reviews['reviewText'].str.len().min(),
        'max_text_length': reviews['reviewText'].str.len().max(),
        'verified_count': reviews['verified'].sum(),
        'verified_ratio': reviews['verified'].mean()
    }
    
    all_reviews_text = ' '.join([str(text) for text in reviews['reviewText'].fillna('')])
    user_features[user_id]['all_reviews_keywords'] = extract_keywords(all_reviews_text)
    
    user_asins = reviews['asin'].unique().tolist()
    
    user_items_data = items_df[items_df['asin'].isin(user_asins)]
    
    if len(user_items_data) > 0:
        brand_counts = user_items_data['brand'].value_counts()
        if len(brand_counts) > 0:
            user_features[user_id]['top_category'] = brand_counts.idxmax()
            user_features[user_id]['top_category_count'] = brand_counts.max()
        else:
            user_features[user_id]['top_category'] = ''
            user_features[user_id]['top_category_count'] = 0
    else:
        user_features[user_id]['top_category'] = ''
        user_features[user_id]['top_category_count'] = 0
    
    all_titles = ' '.join([str(title) for title in user_items_data['title'].fillna('')])
    user_features[user_id]['top_style_colors'] = extract_color_from_text(all_titles)
    user_features[user_id]['top_style_sizes'] = extract_size_from_text(all_titles)

print("Creating DataFrame...")
user_feat_df = pd.DataFrame.from_dict(user_features, orient='index')
user_feat_df = user_feat_df.reset_index()
user_feat_df.rename(columns={'index': 'reviewerID'}, inplace=True)

print("Handling missing values...")
for col in user_feat_df.columns:
    if col == 'reviewerID':
        continue
    
    if user_feat_df[col].dtype in ['float64', 'int64']:
        user_feat_df[col].fillna(user_feat_df[col].mean(), inplace=True)
    else:
        user_feat_df[col].fillna('', inplace=True)

print("Processing CLIP encoding for user reviews...")
user_texts = []
user_ids = []

for user_id, group in reviews_df.groupby('reviewerID'):
    overall_text = ' '.join([str(text) for text in group['reviewText'].fillna('')])
    user_texts.append(overall_text[:77] if len(overall_text) > 77 else overall_text)
    user_ids.append(user_id)

print(f"Encoding {len(user_texts)} user texts...")
content_clip_vectors = []

batch_size = 32
for i in range(0, len(user_texts), batch_size):
    batch_texts = user_texts[i:i+batch_size]
    
    text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    content_clip_vectors.append(text_features.cpu().numpy())
    
    if (i + batch_size) % 100 == 0:
        print(f"Processed {i + batch_size}/{len(user_texts)}")

content_clip_vectors_array = np.vstack(content_clip_vectors)

print(f"Content CLIP vectors shape: {content_clip_vectors_array.shape}")

print("Applying TSNE dimensionality reduction...")
print("Reducing user content vectors to 3D...")
content_tsne_vectors = apply_tsne(content_clip_vectors_array, n_components=3)

print(f"Content TSNE vectors shape: {content_tsne_vectors.shape}")

print("Adding reduced vectors to DataFrame...")
user_feat_df['content_vector'] = [vector_to_list(v) for v in content_tsne_vectors]

print("Saving user features to CSV...")
user_feat_df.to_csv('new_feat/user.csv', index=False, encoding='utf-8')
print(f"Saved {len(user_feat_df)} users to new_feat/user.csv")

print("\n=== User Feature Statistics ===")
print(f"Total users: {len(user_feat_df)}")
print(f"Total features: {len(user_feat_df.columns)}")

print(f"\nRating statistics:")
print(f"  Average rating: {user_feat_df['avg_rating'].mean():.2f}")
print(f"  Rating std: {user_feat_df['rating_std'].mean():.2f}")
print(f"  Min rating: {user_feat_df['min_rating'].mean():.2f}")
print(f"  Max rating: {user_feat_df['max_rating'].mean():.2f}")

print(f"\nReview statistics:")
print(f"  Average review count: {user_feat_df['review_count'].mean():.1f}")
print(f"  Min review count: {user_feat_df['review_count'].min()}")
print(f"  Max review count: {user_feat_df['review_count'].max()}")

print(f"\nText statistics:")
print(f"  Average text length: {user_feat_df['avg_text_length'].mean():.1f}")
print(f"  Text length std: {user_feat_df['text_length_std'].mean():.1f}")

print(f"\nItem preference statistics:")
print(f"  Users with top_category: {len([idx for idx in user_feat_df['top_category'] if idx != ''])}")
print(f"  Users with top_style_colors: {len([idx for idx, colors in zip(user_feat_df['reviewerID'], user_feat_df['top_style_colors']) if len(colors) > 0])}")
print(f"  Users with top_style_sizes: {len([idx for idx, sizes in zip(user_feat_df['reviewerID'], user_feat_df['top_style_sizes']) if len(sizes) > 0])}")

print(f"\nContent TSNE vector statistics:")
print(f"  Shape: {content_tsne_vectors.shape}")
print(f"  Mean: {content_tsne_vectors.mean():.6f}")
print(f"  Std: {content_tsne_vectors.std():.6f}")
print(f"  Min: {content_tsne_vectors.min():.6f}")
print(f"  Max: {content_tsne_vectors.max():.6f}")

print("\n=== Feature columns ===")
print(f"Columns in user.csv: {list(user_feat_df.columns)}")

print("\n=== Sample data ===")
print(f"First user ID: {user_feat_df.iloc[0]['reviewerID']}")
print(f"Content vector (first 3 dims): {user_feat_df.iloc[0]['content_vector']}")
print(f"Review count: {user_feat_df.iloc[0]['review_count']}")
print(f"Average rating: {user_feat_df.iloc[0]['avg_rating']:.2f}")

print("\n=== Completed ===")
print("Generated files:")
print("  - new_feat/user.csv (user features with 3D content vectors stored as lists)")
