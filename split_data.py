"""Split each user's interactions into train/test subsets.

The current project keeps a simple chronological split: the first 70% of a
user's reviews become training interactions and the remainder become test
interactions. The outputs are later used for feature generation and model
training/evaluation.
"""

import numpy as np
import pandas as pd

TRAIN_RATIO = 0.7
DATA_DIR = 'new_dataset'


def main():
    """Generate `user_item.csv` and `train_reviews.csv` from cleaned reviews."""
    print("Loading reviews data...")
    reviews_df = pd.read_csv(f'{DATA_DIR}/reviews.csv')

    print("Processing user data...")
    user_item_list = []
    train_reviews_list = []

    # Build train/test sequences user by user to preserve interaction order.
    for user_id, group in reviews_df.groupby('reviewerID'):
        group = group.sort_values('unixReviewTime').reset_index(drop=True)

        total_reviews = len(group)
        split_idx = int(np.ceil(total_reviews * TRAIN_RATIO))

        train_group = group.iloc[:split_idx]
        test_group = group.iloc[split_idx:]

        train_items = '|'.join(train_group['asin'].astype(str).unique())
        test_items = '|'.join(test_group['asin'].astype(str).unique())

        # The recommendation pipeline expects both train and test items per user.
        if not train_items or not test_items:
            continue

        user_item_list.append({
            'user_id': user_id,
            'train': train_items,
            'test': test_items
        })
        train_reviews_list.append(train_group)

    user_item_df = pd.DataFrame(user_item_list)
    user_item_df.to_csv(f'{DATA_DIR}/user_item.csv', index=False, encoding='utf-8')
    print(f"Saved {len(user_item_df)} users to {DATA_DIR}/user_item.csv")

    train_reviews_df = pd.concat(train_reviews_list, ignore_index=True)
    train_reviews_df.to_csv(f'{DATA_DIR}/train_reviews.csv', index=False, encoding='utf-8')
    print(f"Saved {len(train_reviews_df)} training reviews to {DATA_DIR}/train_reviews.csv")

    print("Data splitting completed!")


if __name__ == '__main__':
    main()
