"""Raw Amazon Fashion dataset preprocessing.

This script reads the original review metadata, filters sparse users,
keeps only interacted items, cleans noisy HTML-like fields, and writes the
intermediate CSV files used by later feature-engineering steps.
"""

import json
import os
import re

import pandas as pd

MIN_REVIEWS_PER_USER = 7
OUTPUT_DIR = 'new_dataset'


def load_jsonl(path):
    """Load a JSON Lines file into a DataFrame."""
    records = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def fill_missing_values(df):
    """Fill numeric columns with means and text columns with empty strings."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna('')
    return df


def clean_html_js(text):
    """Remove script tags, HTML tags, and extra whitespace from text fields."""
    if not isinstance(text, str):
        return ''

    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_item_field(value):
    """Normalize raw item fields so list/text columns can be stored as plain text."""
    if isinstance(value, list):
        return ' '.join(clean_html_js(str(item)) for item in value)
    if isinstance(value, str):
        return clean_html_js(value)
    return ''


def extract_rank(rank_str):
    """Parse Amazon rank strings into a numeric rank and category name."""
    if not isinstance(rank_str, str) or not rank_str:
        return None, None

    match = re.match(r'([\d,]+)\s*in\s*(.+?)(?:\(|$)', rank_str)
    if not match:
        return None, None

    rank_num = match.group(1).replace(',', '')
    rank_field = match.group(2).strip()
    try:
        rank_num = int(rank_num)
    except ValueError:
        rank_num = None
    return rank_num, rank_field


def main():
    """Run the raw-data preprocessing pipeline end to end."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Stage 1: load raw JSONL files.
    print("Loading data...")
    reviews_df = load_jsonl('dataset/AMAZON_FASHION.json')
    items_df = load_jsonl('dataset/meta_AMAZON_FASHION.json')
    items_df = items_df.drop_duplicates(subset=['asin']).reset_index(drop=True)
    print(f"Loaded {len(reviews_df)} reviews and {len(items_df)} items")

    # Stage 2: clean review records and drop users with too few interactions.
    print("Cleaning reviews data...")
    reviews_df = fill_missing_values(reviews_df)

    user_review_counts = reviews_df['reviewerID'].value_counts()
    valid_users = user_review_counts[user_review_counts >= MIN_REVIEWS_PER_USER].index
    reviews_df = reviews_df[reviews_df['reviewerID'].isin(valid_users)].reset_index(drop=True)
    print(f"After filtering users with >= {MIN_REVIEWS_PER_USER} reviews: {len(reviews_df)} reviews")

    # Stage 3: keep only items that still appear after user filtering.
    valid_asins = set(reviews_df['asin'].unique())
    items_df = items_df[items_df['asin'].isin(valid_asins)].reset_index(drop=True)
    print(f"Valid items: {len(items_df)}")

    # Stage 4: clean item-side text and parse ranking metadata.
    print("Cleaning items data...")
    for col in items_df.columns:
        items_df[col] = items_df[col].apply(clean_item_field)

    items_df[['rank_num', 'rank_field']] = items_df['rank'].apply(
        lambda value: pd.Series(extract_rank(value))
    )
    items_df = fill_missing_values(items_df)

    # Stage 5: write the intermediate tabular dataset consumed by split_data.py.
    print("Saving data...")
    review_cols = [
        'reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText', 'overall',
        'summary', 'unixReviewTime', 'reviewTime', 'verified', 'style'
    ]
    review_cols = [col for col in review_cols if col in reviews_df.columns]
    reviews_df[review_cols].to_csv(f'{OUTPUT_DIR}/reviews.csv', index=False, encoding='utf-8')

    item_cols = [
        'asin', 'title', 'brand', 'feature', 'description', 'price', 'imageURL',
        'imageURLHighRes', 'rank', 'rank_num', 'rank_field', 'date', 'also_view',
        'also_buy', 'fit', 'details', 'similar_item', 'tech1'
    ]
    item_cols = [col for col in item_cols if col in items_df.columns]
    items_df[item_cols].to_csv(f'{OUTPUT_DIR}/item.csv', index=False, encoding='utf-8')

    print(f"Saved {len(reviews_df)} reviews to {OUTPUT_DIR}/reviews.csv")
    print(f"Saved {len(items_df)} items to {OUTPUT_DIR}/item.csv")
    print("Preprocessing completed!")


if __name__ == '__main__':
    main()

