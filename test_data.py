import pandas as pd
import numpy as np

print("Loading data...")
user_df = pd.read_csv('new_feat/user.csv')
item_df = pd.read_csv('new_feat/item.csv')
user_item_df = pd.read_csv('new_dataset/user_item.csv')

print(f"Users: {len(user_df)}, Items: {len(item_df)}, User-Item pairs: {len(user_item_df)}")

print("\nUser item data sample:")
print(user_item_df.head(10))

print("\nChecking train column...")
print(f"Train column type: {user_item_df['train'].dtype}")
print(f"Train column null count: {user_item_df['train'].isnull().sum()}")
print(f"Train column sample values:")
print(user_item_df['train'].head(10))

print("\nChecking test column...")
print(f"Test column type: {user_item_df['test'].dtype}")
print(f"Test column null count: {user_item_df['test'].isnull().sum()}")
print(f"Test column sample values:")
print(user_item_df['test'].head(10))

print("\nParsing train items...")
train_items_list = []
for idx, row in user_item_df.iterrows():
    if idx < 10:
        train_str = row['train']
        if pd.notna(train_str):
            train_items = train_str.split('|')
            train_items_list.append(len(train_items))
            print(f"Row {idx}: {len(train_items)} items")
        else:
            train_items_list.append(0)
            print(f"Row {idx}: NaN")

print(f"\nAverage train items: {np.mean(train_items_list)}")
