import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import ast
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgcn_model import MultiModalLightGCN
    print("Model import successful!")
except Exception as e:
    print(f"Model import failed: {e}")
    import traceback
    traceback.print_exc()

print("Loading data...")
try:
    user_df = pd.read_csv('new_feat/user.csv')
    item_df = pd.read_csv('new_feat/item.csv')
    user_item_df = pd.read_csv('new_dataset/user_item.csv')
    print(f"Users: {len(user_df)}, Items: {len(item_df)}, User-Item pairs: {len(user_item_df)}")
except Exception as e:
    print(f"Data loading failed: {e}")
    import traceback
    traceback.print_exc()
