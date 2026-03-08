import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import ast
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_dir='new_feat', dataset_dir='new_dataset'):
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        self.all_colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 
                         'orange', 'brown', 'gray', 'silver', 'gold', 'beige', 'cream', 
                         'navy', 'tan', 'khaki', 'maroon', 'olive', 'teal', 'charcoal']
        
        self.all_sizes = ['small', 'medium', 'large', 'x-small', 'x-large', 'xx-small', 'xx-large',
                        'one size', '2xl', '3xl', '4xl', '5xl', 'plus size', 'xs', 's', 'm', 'l', 'xl']
        
    def load_data(self):
        print("Loading data...")
        self.user_df = pd.read_csv(f'{self.data_dir}/user.csv')
        self.item_df = pd.read_csv(f'{self.data_dir}/item.csv')
        self.user_item_df = pd.read_csv(f'{self.dataset_dir}/user_item.csv')
        
        print(f"Users: {len(self.user_df)}, Items: {len(self.item_df)}, User-Item pairs: {len(self.user_item_df)}")
        
        return self.user_df, self.item_df, self.user_item_df
    
    def encode_ids(self):
        print("Encoding user and item IDs...")
        all_user_ids = pd.concat([self.user_df['reviewerID'], self.user_item_df['user_id']]).unique()
        all_item_ids = pd.concat([self.item_df['asin'], self.user_item_df['train'].str.split('|').explode(), 
                                       self.user_item_df['test'].str.split('|').explode()]).unique()
        
        self.user_encoder.fit(all_user_ids)
        self.item_encoder.fit(all_item_ids)
        
        self.user_df['user_idx'] = self.user_encoder.transform(self.user_df['reviewerID'])
        self.item_df['item_idx'] = self.item_encoder.transform(self.item_df['asin'])
        
        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_)
        
        print(f"Num users: {self.num_users}, Num items: {self.num_items}")
        
        return self.num_users, self.num_items
    
    @staticmethod
    def parse_vector_string(vector_str):
        try:
            vec = np.array(ast.literal_eval(vector_str), dtype=np.float32)
            return vec
        except:
            return np.zeros(3, dtype=np.float32)
    
    @staticmethod
    def parse_list_string(list_str):
        try:
            return ast.literal_eval(list_str)
        except:
            return []
    
    def parse_features(self):
        print("Parsing features...")
        
        self.user_df['content_vector'] = self.user_df['content_vector'].apply(self.parse_vector_string)
        self.user_df['top_style_colors'] = self.user_df['top_style_colors'].apply(self.parse_list_string)
        self.user_df['top_style_sizes'] = self.user_df['top_style_sizes'].apply(self.parse_list_string)
        
        self.item_df['title_vector'] = self.item_df['title_vector'].apply(self.parse_vector_string)
        self.item_df['image_vector'] = self.item_df['image_vector'].apply(self.parse_vector_string)
        self.item_df['feature_vector'] = self.item_df['feature_vector'].apply(self.parse_vector_string)
        self.item_df['description_vector'] = self.item_df['description_vector'].apply(self.parse_vector_string)
        
        return self.user_df, self.item_df
    
    def extract_color_index(self, color_list):
        if not color_list:
            return 0
        return self.all_colors.index(color_list[0]) if color_list[0] in self.all_colors else 0
    
    def extract_size_index(self, size_list):
        if not size_list:
            return 0
        return self.all_sizes.index(size_list[0]) if size_list[0] in self.all_sizes else 0
    
    @staticmethod
    def extract_price(price_str):
        try:
            prices = [float(p) for p in str(price_str).split('-') if p.replace('.', '').replace(',', '').isdigit()]
            return np.mean(prices) if prices else 0.0
        except:
            return 0.0
    
    def create_user_features(self):
        print("Creating user features...")
        
        user_color_indices = np.array([self.extract_color_index(colors) for colors in self.user_df['top_style_colors']])
        user_size_indices = np.array([self.extract_size_index(sizes) for sizes in self.user_df['top_style_sizes']])
        
        user_numeric_features = np.column_stack([
            self.user_df['review_count'].fillna(0).values,
            self.user_df['avg_rating'].fillna(0).values,
            self.user_df['rating_std'].fillna(0).values,
            self.user_df['min_rating'].fillna(0).values,
            self.user_df['max_rating'].fillna(0).values,
            self.user_df['avg_text_length'].fillna(0).values,
            self.user_df['text_length_std'].fillna(0).values,
            self.user_df['min_text_length'].fillna(0).values,
            self.user_df['max_text_length'].fillna(0).values,
            self.user_df['verified_count'].fillna(0).values,
            self.user_df['verified_ratio'].fillna(0).values,
            self.user_df['top_category_count'].fillna(0).values
        ]).astype(np.float32)
        
        user_vector_features = np.array(self.user_df['content_vector'].tolist())
        
        user_features = np.column_stack([
            user_numeric_features,
            user_vector_features
        ])
        
        print(f"User features shape: {user_features.shape}")
        
        return user_features, user_color_indices, user_size_indices
    
    def create_item_features(self):
        print("Creating item features...")
        
        self.item_df['price_numeric'] = self.item_df['price'].apply(self.extract_price)
        
        item_numeric_features = np.column_stack([
            self.item_df['rank_num'].fillna(0).values,
            self.item_df['rank_num'].fillna(0).values / (self.item_df['rank_num'].max() + 1),
            self.item_df['price_numeric'].fillna(0).values,
            self.item_df['price_numeric'].fillna(0).values / (self.item_df['price_numeric'].max() + 1),
            (self.item_df['also_view'].str.count('\|') + 1).fillna(0).values
        ]).astype(np.float32)
        
        item_vector_features = np.column_stack([
            np.array(self.item_df['title_vector'].tolist()),
            np.array(self.item_df['image_vector'].tolist()),
            np.array(self.item_df['feature_vector'].tolist()),
            np.array(self.item_df['description_vector'].tolist())
        ])
        
        item_features = np.column_stack([
            item_numeric_features,
            item_vector_features
        ])
        
        print(f"Item features shape: {item_features.shape}")
        
        return item_features
    
    def scale_features(self, user_features, item_features):
        print("Scaling features...")
        scaler = StandardScaler()
        user_features_scaled = scaler.fit_transform(user_features).astype(np.float32)
        item_features_scaled = scaler.fit_transform(item_features).astype(np.float32)
        
        return user_features_scaled, item_features_scaled
    
    def create_tensors(self, user_features, item_features, user_color_indices, user_size_indices, device):
        print("Creating tensors...")
        
        user_features_tensor = torch.FloatTensor(user_features).to(device)
        user_color_indices_tensor = torch.LongTensor(user_color_indices).to(device)
        user_size_indices_tensor = torch.LongTensor(user_size_indices).to(device)
        
        valid_item_ids = set(self.item_encoder.classes_)
        valid_item_features = []
        for item_id in self.item_encoder.classes_:
            item_idx = self.item_df[self.item_df['asin'] == item_id].index
            if len(item_idx) > 0:
                valid_item_features.append(item_features[item_idx[0]])
        
        item_features_tensor = torch.FloatTensor(np.array(valid_item_features)).to(device)
        
        print(f"User features tensor shape: {user_features_tensor.shape}")
        print(f"Item features tensor shape: {item_features_tensor.shape}")
        
        return user_features_tensor, user_color_indices_tensor, user_size_indices_tensor, item_features_tensor
    
    def build_interaction_matrix(self):
        print("Building interaction matrix...")
        
        valid_item_ids = set(self.item_encoder.classes_)
        interaction_matrix = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        
        for idx, row in self.user_item_df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing edge {idx}/{len(self.user_item_df)}")
            
            user_id = row['user_id']
            train_items = row['train'].split('|') if pd.notna(row['train']) else []
            
            if user_id not in self.user_encoder.classes_:
                continue
            
            user_idx = self.user_encoder.transform([user_id])[0]
            valid_train_items = [item_id for item_id in train_items if item_id in valid_item_ids]
            
            for item_id in valid_train_items:
                item_idx = self.item_encoder.transform([item_id])[0]
                interaction_matrix[user_idx, item_idx] = 1.0
        
        print(f"Interaction matrix shape: {interaction_matrix.shape}")
        print(f"Interaction matrix density: {np.sum(interaction_matrix) / (self.num_users * self.num_items):.4f}")
        
        return interaction_matrix
    
    def build_graph_edges(self):
        print("Building graph edges...")
        
        valid_item_ids = set(self.item_encoder.classes_)
        user_item_edges = []
        item_user_edges = []
        
        for idx, row in self.user_item_df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing edge {idx}/{len(self.user_item_df)}")
            
            user_id = row['user_id']
            train_items = row['train'].split('|') if pd.notna(row['train']) else []
            
            if user_id not in self.user_encoder.classes_:
                continue
            
            user_idx = self.user_encoder.transform([user_id])[0]
            valid_train_items = [item_id for item_id in train_items if item_id in valid_item_ids]
            
            for item_id in valid_train_items:
                item_idx = self.item_encoder.transform([item_id])[0]
                user_item_edges.append([user_idx, item_idx + self.num_users])
                item_user_edges.append([item_idx + self.num_users, user_idx])
        
        edge_index = torch.tensor(np.array(user_item_edges + item_user_edges), dtype=torch.long).t().contiguous()
        user_item_edge_index = torch.tensor(np.array(user_item_edges), dtype=torch.long).t().contiguous()
        
        print(f"Edge index shape: {edge_index.shape}")
        print(f"User-Item edge index shape: {user_item_edge_index.shape}")
        
        return edge_index, user_item_edge_index
    
    def split_data(self):
        print("Splitting data...")
        
        train_users, test_users = train_test_split(self.user_item_df['user_id'].unique(), test_size=0.2, random_state=42)
        
        train_user_item_df = self.user_item_df[self.user_item_df['user_id'].isin(train_users)]
        test_user_item_df = self.user_item_df[self.user_item_df['user_id'].isin(test_users)]
        
        print(f"Train users: {len(train_users)}, Test users: {len(test_users)}")
        
        return train_user_item_df, test_user_item_df
    
    def prepare_test_data(self, test_user_item_df):
        print("Preparing test data...")
        
        test_users_list = []
        test_items_list = []
        total_test_items = 0
        
        for idx, row in test_user_item_df.iterrows():
            user_id = row['user_id']
            test_items = row['test'].split('|') if pd.notna(row['test']) else []
            
            if user_id not in self.user_encoder.classes_:
                continue
            
            user_idx = self.user_encoder.transform([user_id])[0]
            valid_test_items = [item for item in test_items if item in self.item_encoder.classes_]
            
            if len(valid_test_items) == 0:
                continue
            
            total_test_items += len(valid_test_items)
            test_item_indices = set(self.item_encoder.transform(valid_test_items))
            
            test_users_list.append(user_idx)
            test_items_list.append(test_item_indices)
        
        print(f"Total test users: {len(test_users_list)}, Total test items: {total_test_items}")
        
        return test_users_list, test_items_list, total_test_items
    
    def preprocess_all(self, device):
        self.load_data()
        self.encode_ids()
        self.parse_features()
        
        user_features, user_color_indices, user_size_indices = self.create_user_features()
        item_features = self.create_item_features()
        
        user_features_scaled, item_features_scaled = self.scale_features(user_features, item_features)
        
        user_features_tensor, user_color_indices_tensor, user_size_indices_tensor, item_features_tensor = \
            self.create_tensors(user_features_scaled, item_features_scaled, user_color_indices, user_size_indices, device)
        
        interaction_matrix = self.build_interaction_matrix()
        edge_index, user_item_edge_index = self.build_graph_edges()
        
        train_user_item_df, test_user_item_df = self.split_data()
        test_users_list, test_items_list, total_test_items = self.prepare_test_data(test_user_item_df)
        
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'user_features_tensor': user_features_tensor,
            'user_color_indices_tensor': user_color_indices_tensor,
            'user_size_indices_tensor': user_size_indices_tensor,
            'item_features_tensor': item_features_tensor,
            'interaction_matrix': interaction_matrix,
            'edge_index': edge_index,
            'user_item_edge_index': user_item_edge_index,
            'train_user_item_df': train_user_item_df,
            'test_user_item_df': test_user_item_df,
            'test_users_list': test_users_list,
            'test_items_list': test_items_list,
            'total_test_items': total_test_items,
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder
        }