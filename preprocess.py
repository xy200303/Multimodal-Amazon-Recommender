import json
import pandas as pd
import numpy as np
import re
import os
from collections import Counter

# 创建输出目录
os.makedirs('new_dataset', exist_ok=True)

# 1. 加载数据
print("Loading data...")
reviews_list = []
with open('dataset/AMAZON_FASHION.json', 'r', encoding='utf-8') as f:
    for line in f:
        reviews_list.append(json.loads(line))

items_list = []
with open('dataset/meta_AMAZON_FASHION.json', 'r', encoding='utf-8') as f:
    for line in f:
        items_list.append(json.loads(line))

reviews_df = pd.DataFrame(reviews_list)
items_df = pd.DataFrame(items_list)

print(f"Loaded {len(reviews_df)} reviews and {len(items_df)} items")

# 2. 清洗评论数据
print("Cleaning reviews data...")

# 处理缺失值
for col in reviews_df.columns:
    if reviews_df[col].dtype in ['float64', 'int64']:
        reviews_df[col].fillna(reviews_df[col].mean(), inplace=True)
    else:
        reviews_df[col].fillna('', inplace=True)

# 3. 过滤用户：保留评论数 >= 5 的用户
user_review_counts = reviews_df['reviewerID'].value_counts()
valid_users = user_review_counts[user_review_counts >= 5].index
reviews_df = reviews_df[reviews_df['reviewerID'].isin(valid_users)].reset_index(drop=True)

print(f"After filtering users with >= 5 reviews: {len(reviews_df)} reviews")

# 4. 获取有效的商品ID
valid_asins = set(reviews_df['asin'].unique())
items_df = items_df[items_df['asin'].isin(valid_asins)].reset_index(drop=True)

print(f"Valid items: {len(items_df)}")

# 5. 清洗商品数据
print("Cleaning items data...")

def clean_html_js(text):
    """清洗包含HTML和JS源码的字段"""
    if not isinstance(text, str):
        return ''
    # 移除JS代码
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # 移除HTML标签（包括双引号的情况）
    text = re.sub(r'<[^>]+>', '', text)
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else ''

def extract_rank(rank_str):
    """从rank字段提取rank_num和rank_field"""
    if not isinstance(rank_str, str) or not rank_str:
        return None, None

    # 匹配格式: "13,052,976inClothing,Shoesamp;Jewelry("
    match = re.match(r'([\d,]+)\s*in\s*(.+?)(?:\(|$)', rank_str)
    if match:
        rank_num = match.group(1).replace(',', '')
        rank_field = match.group(2).strip()
        try:
            rank_num = int(rank_num)
        except:
            rank_num = None
        return rank_num, rank_field
    return None, None

# 清洗字段 - 对所有字符串字段进行HTML清洗
for col in items_df.columns:
    if len(items_df) > 0:
        def clean_field(x):
            if isinstance(x, list):
                return ' '.join([clean_html_js(str(i)) for i in x])
            elif isinstance(x, str):
                return clean_html_js(x)
            else:
                return ''
        items_df[col] = items_df[col].apply(clean_field)

# 提取rank字段
items_df[['rank_num', 'rank_field']] = items_df['rank'].apply(lambda x: pd.Series(extract_rank(x)))

# 处理缺失值
for col in items_df.columns:
    if items_df[col].dtype in ['float64', 'int64']:
        items_df[col].fillna(items_df[col].mean(), inplace=True)
    else:
        items_df[col].fillna('', inplace=True)

# 6. 输出数据
print("Saving data...")

# 评论字段
review_cols = ['reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText', 'overall',
               'summary', 'unixReviewTime', 'reviewTime', 'verified', 'style']
review_cols = [col for col in review_cols if col in reviews_df.columns]
reviews_df[review_cols].to_csv('new_dataset/reviews.csv', index=False, encoding='utf-8')

# 商品字段
item_cols = ['asin', 'title', 'brand', 'feature', 'description', 'price', 'imageURL',
             'imageURLHighRes', 'rank', 'rank_num', 'rank_field', 'date', 'also_view',
             'also_buy', 'fit', 'details', 'similar_item', 'tech1']
item_cols = [col for col in item_cols if col in items_df.columns]
items_df[item_cols].to_csv('new_dataset/item.csv', index=False, encoding='utf-8')

print(f"Saved {len(reviews_df)} reviews to new_dataset/reviews.csv")
print(f"Saved {len(items_df)} items to new_dataset/item.csv")
print("Preprocessing completed!")
