import pandas as pd
import numpy as np

# 读取数据
print("Loading reviews data...")
reviews_df = pd.read_csv('new_dataset/reviews.csv')

# 按reviewerID分组，对每个用户的评论按时间排序
print("Processing user data...")
user_item_list = []
train_reviews_list = []

for user_id, group in reviews_df.groupby('reviewerID'):
    # 按unixReviewTime排序
    group = group.sort_values('unixReviewTime').reset_index(drop=True)

    # 计算70%和30%的分割点
    total_reviews = len(group)
    split_idx = int(np.ceil(total_reviews * 0.7))

    # 分割训练集和测试集
    train_group = group.iloc[:split_idx]
    test_group = group.iloc[split_idx:]

    # 获取商品ID列表
    train_items = '|'.join(train_group['asin'].astype(str).unique())
    test_items = '|'.join(test_group['asin'].astype(str).unique())

    # 添加到user_item列表
    if train_items and test_items:  # 只保留既有训练集又有测试集的用户
        user_item_list.append({
            'user_id': user_id,
            'train': train_items,
            'test': test_items
        })

        # 添加训练集评论
        train_reviews_list.append(train_group)

# 生成user_item.csv
user_item_df = pd.DataFrame(user_item_list)
user_item_df.to_csv('new_dataset/user_item.csv', index=False, encoding='utf-8')
print(f"Saved {len(user_item_df)} users to new_dataset/user_item.csv")

# 生成train_reviews.csv
train_reviews_df = pd.concat(train_reviews_list, ignore_index=True)
train_reviews_df.to_csv('new_dataset/train_reviews.csv', index=False, encoding='utf-8')
print(f"Saved {len(train_reviews_df)} training reviews to new_dataset/train_reviews.csv")

print("Data splitting completed!")
