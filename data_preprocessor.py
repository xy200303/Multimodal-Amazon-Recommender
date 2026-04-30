"""将离线特征文件转换为训练所需的张量与图结构。

该模块连接离线预处理结果与 PyTorch 训练/评测流程，主要负责：
1. 用户与商品 ID 编码；
2. 解析 CSV 中保存的向量和列表字面量；
3. 构建用户/商品特征张量；
4. 构建交互矩阵与图边结构；
5. 整理测试阶段所需的目标商品集合。
"""

import pandas as pd
import numpy as np
import torch
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import ast
import warnings
warnings.filterwarnings('ignore')

VECTOR_FALLBACK_DIM = 64

ITEM_ABLATION_MODES = {
    'all_features',
    'numeric_text',
    'text_image'
}


class DataPreprocessor:
    def __init__(self, data_dir='new_feat', dataset_dir='new_dataset', ablation_mode='all_features'):
        """设置数据路径与颜色/尺码等离散特征词表。"""
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.ablation_mode = ablation_mode
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        self.all_colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 
                         'orange', 'brown', 'gray', 'silver', 'gold', 'beige', 'cream', 
                         'navy', 'tan', 'khaki', 'maroon', 'olive', 'teal', 'charcoal']
        
        self.all_sizes = ['small', 'medium', 'large', 'x-small', 'x-large', 'xx-small', 'xx-large',
                        'one size', '2xl', '3xl', '4xl', '5xl', 'plus size', 'xs', 's', 'm', 'l', 'xl']
        self.num_colors = len(self.all_colors) + 1
        self.num_sizes = len(self.all_sizes) + 1

        if self.ablation_mode not in ITEM_ABLATION_MODES:
            valid_modes = ', '.join(sorted(ITEM_ABLATION_MODES))
            raise ValueError(f"Unknown ablation_mode: {self.ablation_mode}. Valid values: {valid_modes}")
        
    def load_data(self):
        """读取用户特征、商品特征和训练/测试切分结果。"""
        print("Loading data...")
        self.user_df = pd.read_csv(f'{self.data_dir}/user.csv')
        self.item_df = pd.read_csv(f'{self.data_dir}/item.csv')
        self.user_item_df = pd.read_csv(f'{self.dataset_dir}/user_item.csv')
        
        print(f"Users: {len(self.user_df)}, Items: {len(self.item_df)}, User-Item pairs: {len(self.user_item_df)}")
        
        return self.user_df, self.item_df, self.user_item_df
    
    def encode_ids(self):
        """将原始用户/商品标识映射为连续整数索引。"""
        print("Encoding user and item IDs...")
        all_user_ids = pd.concat([self.user_df['reviewerID'], self.user_item_df['user_id']]).unique()
        all_item_ids = pd.concat([self.item_df['asin'], self.user_item_df['train'].str.split('|').explode(), 
                                       self.user_item_df['test'].str.split('|').explode()]).unique()
        
        self.user_encoder.fit(all_user_ids)
        self.item_encoder.fit(all_item_ids)
        
        self.user_df['user_idx'] = self.user_encoder.transform(self.user_df['reviewerID'])
        self.item_df['item_idx'] = self.item_encoder.transform(self.item_df['asin'])
        self.user_id_to_idx = dict(zip(self.user_encoder.classes_, range(len(self.user_encoder.classes_))))
        self.item_id_to_idx = dict(zip(self.item_encoder.classes_, range(len(self.item_encoder.classes_))))
        
        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_)
        
        print(f"Num users: {self.num_users}, Num items: {self.num_items}")
        
        return self.num_users, self.num_items
    
    @staticmethod
    def parse_vector_string(vector_str, fallback_dim=VECTOR_FALLBACK_DIM):
        """解析 CSV 中字符串形式的向量；失败时按目标维度回退为零向量。"""
        try:
            vec = np.array(ast.literal_eval(vector_str), dtype=np.float32)
            return vec
        except:
            return np.zeros(fallback_dim, dtype=np.float32)
    
    @staticmethod
    def parse_list_string(list_str):
        """解析颜色偏好、尺码偏好等列表字段。"""
        try:
            return ast.literal_eval(list_str)
        except:
            return []
    
    def parse_features(self):
        """Deserialize vector/list fields after loading CSV files."""
        print("Parsing features...")
        
        # 按当前离线特征配置解析向量字段，确保 64 维 PCA 向量在异常情况下也能正确补零。
        self.user_df['content_vector'] = self.user_df['content_vector'].apply(
            lambda value: self.parse_vector_string(value, fallback_dim=VECTOR_FALLBACK_DIM)
        )
        self.user_df['top_style_colors'] = self.user_df['top_style_colors'].apply(self.parse_list_string)
        self.user_df['top_style_sizes'] = self.user_df['top_style_sizes'].apply(self.parse_list_string)
        
        self.item_df['title_vector'] = self.item_df['title_vector'].apply(
            lambda value: self.parse_vector_string(value, fallback_dim=VECTOR_FALLBACK_DIM)
        )
        self.item_df['image_vector'] = self.item_df['image_vector'].apply(
            lambda value: self.parse_vector_string(value, fallback_dim=VECTOR_FALLBACK_DIM)
        )
        self.item_df['feature_vector'] = self.item_df['feature_vector'].apply(
            lambda value: self.parse_vector_string(value, fallback_dim=VECTOR_FALLBACK_DIM)
        )
        self.item_df['description_vector'] = self.item_df['description_vector'].apply(
            lambda value: self.parse_vector_string(value, fallback_dim=VECTOR_FALLBACK_DIM)
        )
        
        return self.user_df, self.item_df
    
    def extract_color_index(self, color_list):
        """Map the first preferred color token to a fixed embedding index."""
        if not color_list:
            return 0
        return self.all_colors.index(color_list[0]) + 1 if color_list[0] in self.all_colors else 0
    
    def extract_size_index(self, size_list):
        """Map the first preferred size token to a fixed embedding index."""
        if not size_list:
            return 0
        return self.all_sizes.index(size_list[0]) + 1 if size_list[0] in self.all_sizes else 0

    @staticmethod
    def extract_price(price_str):
        """Convert price strings or ranges to a single numeric value."""
        try:
            prices = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', str(price_str))
            prices = [float(price.replace(',', '')) for price in prices]
            return np.mean(prices) if prices else 0.0
        except:
            return 0.0

    @staticmethod
    def count_delimited_values(value, delimiter='|'):
        """统计以固定分隔符拼接的列表字段长度。"""
        if not isinstance(value, str):
            return 0.0

        tokens = [token.strip() for token in value.split(delimiter) if token.strip()]
        return float(len(tokens))

    @staticmethod
    def measure_text_length(text):
        """计算文本字段长度。"""
        if not isinstance(text, str):
            return 0.0
        return float(len(text.strip()))
    
    def create_user_features(self):
        """Assemble user numeric features, reduced text vectors, and style indices."""
        print("Creating user features...")
        
        user_color_indices = np.array([self.extract_color_index(colors) for colors in self.user_df['top_style_colors']])
        user_size_indices = np.array([self.extract_size_index(sizes) for sizes in self.user_df['top_style_sizes']])
        
        color_count_series = (
            self.user_df['top_style_color_count'].fillna(0).values
            if 'top_style_color_count' in self.user_df.columns
            else self.user_df['top_style_colors'].apply(len).values
        )
        size_count_series = (
            self.user_df['top_style_size_count'].fillna(0).values
            if 'top_style_size_count' in self.user_df.columns
            else self.user_df['top_style_sizes'].apply(len).values
        )

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
            self.user_df['top_category_count'].fillna(0).values,
            color_count_series,
            size_count_series
        ]).astype(np.float32)
        
        user_vector_features = np.array(self.user_df['content_vector'].tolist())
        self.user_numeric_dim = user_numeric_features.shape[1]
        self.user_vector_dim = user_vector_features.shape[1]
        
        user_features = np.column_stack([
            user_numeric_features,
            user_vector_features
        ])
        
        print(f"User features shape: {user_features.shape}")
        
        return user_features, user_color_indices, user_size_indices
    
    def create_item_features(self):
        """Assemble item numeric features and concatenated multimodal vectors."""
        print("Creating item features...")
        
        self.item_df['price_numeric'] = self.item_df['price'].apply(self.extract_price)
        max_rank = self.item_df['rank_num'].fillna(0).max() + 1
        max_price = self.item_df['price_numeric'].fillna(0).max() + 1

        also_view_count = (
            self.item_df['also_view_count'].fillna(0).values
            if 'also_view_count' in self.item_df.columns
            else self.item_df['also_view'].apply(self.count_delimited_values).values
        )
        also_buy_count = (
            self.item_df['also_buy_count'].fillna(0).values
            if 'also_buy_count' in self.item_df.columns
            else self.item_df['also_buy'].apply(self.count_delimited_values).values
        )
        title_length = (
            self.item_df['title_length'].fillna(0).values
            if 'title_length' in self.item_df.columns
            else self.item_df['title'].apply(self.measure_text_length).values
        )
        feature_length = (
            self.item_df['feature_length'].fillna(0).values
            if 'feature_length' in self.item_df.columns
            else self.item_df['feature'].apply(self.measure_text_length).values
        )
        description_length = (
            self.item_df['description_length'].fillna(0).values
            if 'description_length' in self.item_df.columns
            else self.item_df['description'].apply(self.measure_text_length).values
        )
        has_price = (
            self.item_df['has_price'].fillna(0).values
            if 'has_price' in self.item_df.columns
            else self.item_df['price'].fillna('').astype(str).str.strip().ne('').astype(np.float32).values
        )
        has_feature = (
            self.item_df['has_feature'].fillna(0).values
            if 'has_feature' in self.item_df.columns
            else self.item_df['feature'].fillna('').astype(str).str.strip().ne('').astype(np.float32).values
        )
        has_description = (
            self.item_df['has_description'].fillna(0).values
            if 'has_description' in self.item_df.columns
            else self.item_df['description'].fillna('').astype(str).str.strip().ne('').astype(np.float32).values
        )
        has_image = (
            self.item_df['has_image'].fillna(0).values
            if 'has_image' in self.item_df.columns
            else (
                self.item_df['imageURLHighRes'].fillna('').astype(str).str.strip().ne('') |
                self.item_df['imageURL'].fillna('').astype(str).str.strip().ne('')
            ).astype(np.float32).values
        )
        
        item_numeric_features = np.column_stack([
            self.item_df['rank_num'].fillna(0).values,
            self.item_df['rank_num'].fillna(0).values / max_rank,
            self.item_df['price_numeric'].fillna(0).values,
            self.item_df['price_numeric'].fillna(0).values / max_price,
            also_view_count,
            also_buy_count,
            title_length,
            feature_length,
            description_length,
            has_price,
            has_feature,
            has_description,
            has_image
        ]).astype(np.float32)
        
        item_vector_features = np.column_stack([
            np.array(self.item_df['title_vector'].tolist()),
            np.array(self.item_df['image_vector'].tolist()),
            np.array(self.item_df['feature_vector'].tolist()),
            np.array(self.item_df['description_vector'].tolist())
        ])
        self.item_numeric_dim = item_numeric_features.shape[1]
        self.item_vector_dim = item_vector_features.shape[1]
        vector_start = self.item_numeric_dim
        modality_width = self.item_vector_dim // 4
        self.item_feature_slices = {
            'numeric': slice(0, self.item_numeric_dim),
            'title_text': slice(vector_start, vector_start + modality_width),
            'image': slice(vector_start + modality_width, vector_start + modality_width * 2),
            'feature_text': slice(vector_start + modality_width * 2, vector_start + modality_width * 3),
            'description_text': slice(vector_start + modality_width * 3, vector_start + modality_width * 4)
        }
        
        item_features = np.column_stack([
            item_numeric_features,
            item_vector_features
        ])
        
        print(f"Item features shape: {item_features.shape}")
        
        return item_features

    def apply_item_feature_ablation(self, item_features):
        """按预设消融模式屏蔽商品特征的部分模态，保持输入维度不变。"""
        if self.ablation_mode == 'all_features':
            return item_features

        masked_features = item_features.copy()

        if self.ablation_mode == 'numeric_text':
            masked_features[:, self.item_feature_slices['image']] = 0.0
        elif self.ablation_mode == 'text_image':
            masked_features[:, self.item_feature_slices['numeric']] = 0.0

        print(f"Applied item feature ablation mode: {self.ablation_mode}")
        return masked_features
    
    def scale_features(self, user_features, item_features):
        """Normalize dense user/item features before turning them into tensors."""
        print("Scaling features...")
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        user_features_scaled = self.user_scaler.fit_transform(user_features).astype(np.float32)
        item_features_scaled = self.item_scaler.fit_transform(item_features).astype(np.float32)
        
        return user_features_scaled, item_features_scaled
    
    def create_tensors(self, user_features, item_features, user_color_indices, user_size_indices, device):
        """Create device-ready tensors aligned with encoded user/item indices."""
        print("Creating tensors...")
        
        user_features_tensor = torch.FloatTensor(user_features).to(device)
        user_color_indices_tensor = torch.LongTensor(user_color_indices).to(device)
        user_size_indices_tensor = torch.LongTensor(user_size_indices).to(device)
        
        # Item features must follow item_idx order because models index them directly by item id.
        ordered_item_df = self.item_df.sort_values('item_idx')
        ordered_item_features = item_features[ordered_item_df.index.to_numpy()]
        item_features_tensor = torch.FloatTensor(ordered_item_features).to(device)
        
        print(f"User features tensor shape: {user_features_tensor.shape}")
        print(f"Item features tensor shape: {item_features_tensor.shape}")
        
        return user_features_tensor, user_color_indices_tensor, user_size_indices_tensor, item_features_tensor
    
    def build_interaction_matrix(self):
        """Build the binary train interaction matrix used for masking and negative sampling."""
        print("Building interaction matrix...")
        
        interaction_matrix = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        
        for idx, row in self.user_item_df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing edge {idx}/{len(self.user_item_df)}")
            
            user_id = row['user_id']
            train_items = row['train'].split('|') if pd.notna(row['train']) else []
            
            user_idx = self.user_id_to_idx.get(user_id)
            if user_idx is None:
                continue

            for item_id in train_items:
                item_idx = self.item_id_to_idx.get(item_id)
                if item_idx is None:
                    continue
                interaction_matrix[user_idx, item_idx] = 1.0
        
        print(f"Interaction matrix shape: {interaction_matrix.shape}")
        print(f"Interaction matrix density: {np.sum(interaction_matrix) / (self.num_users * self.num_items):.4f}")
        
        return interaction_matrix
    
    def build_graph_edges(self):
        """Build user-item graph edges for LightGCN-style models."""
        print("Building graph edges...")
        
        user_item_edges = []
        item_user_edges = []
        
        for idx, row in self.user_item_df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing edge {idx}/{len(self.user_item_df)}")
            
            user_id = row['user_id']
            train_items = row['train'].split('|') if pd.notna(row['train']) else []
            
            user_idx = self.user_id_to_idx.get(user_id)
            if user_idx is None:
                continue

            for item_id in train_items:
                item_idx = self.item_id_to_idx.get(item_id)
                if item_idx is None:
                    continue
                user_item_edges.append([user_idx, item_idx + self.num_users])
                item_user_edges.append([item_idx + self.num_users, user_idx])
        
        edge_index = torch.tensor(np.array(user_item_edges + item_user_edges), dtype=torch.long).t().contiguous()
        user_item_edge_index = torch.tensor(np.array(user_item_edges), dtype=torch.long).t().contiguous()
        
        print(f"Edge index shape: {edge_index.shape}")
        print(f"User-Item edge index shape: {user_item_edge_index.shape}")
        
        return edge_index, user_item_edge_index
    
    def split_data(self):
        """Return the precomputed train/test split files without creating a validation split."""
        print("Splitting data...")
        
        train_user_item_df = self.user_item_df.copy()
        test_user_item_df = self.user_item_df.copy()
        
        print(f"All users will be evaluated: {test_user_item_df['user_id'].nunique()}")
        
        return train_user_item_df, test_user_item_df
    
    def prepare_test_data(self, test_user_item_df):
        """Convert each test user's ground-truth items to encoded index sets."""
        print("Preparing test data...")
        
        test_users_list = []
        test_items_list = []
        total_test_items = 0
        
        for idx, row in test_user_item_df.iterrows():
            user_id = row['user_id']
            test_items = row['test'].split('|') if pd.notna(row['test']) else []
            
            user_idx = self.user_id_to_idx.get(user_id)
            if user_idx is None:
                continue

            valid_test_items = [self.item_id_to_idx[item] for item in test_items if item in self.item_id_to_idx]
            
            if not valid_test_items:
                continue
            
            total_test_items += len(valid_test_items)
            test_item_indices = set(valid_test_items)
            
            test_users_list.append(user_idx)
            test_items_list.append(test_item_indices)
        
        print(f"Total test users: {len(test_users_list)}, Total test items: {total_test_items}")
        
        return test_users_list, test_items_list, total_test_items
    
    def preprocess_all(self, device):
        """Run the full tensor/graph preparation pipeline and return a training dictionary."""
        self.load_data()
        self.encode_ids()
        self.parse_features()
        
        user_features, user_color_indices, user_size_indices = self.create_user_features()
        item_features = self.create_item_features()
        
        user_features_scaled, item_features_scaled = self.scale_features(user_features, item_features)
        item_features_scaled = self.apply_item_feature_ablation(item_features_scaled)
        
        user_features_tensor, user_color_indices_tensor, user_size_indices_tensor, item_features_tensor = \
            self.create_tensors(user_features_scaled, item_features_scaled, user_color_indices, user_size_indices, device)
        
        interaction_matrix = self.build_interaction_matrix()
        edge_index, user_item_edge_index = self.build_graph_edges()
        edge_index = edge_index.to(device)
        user_item_edge_index = user_item_edge_index.to(device)
        
        train_user_item_df, test_user_item_df = self.split_data()
        test_users_list, test_items_list, total_test_items = self.prepare_test_data(test_user_item_df)
        
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_colors': self.num_colors,
            'num_sizes': self.num_sizes,
            'user_numeric_dim': self.user_numeric_dim,
            'user_vector_dim': self.user_vector_dim,
            'item_numeric_dim': self.item_numeric_dim,
            'item_vector_dim': self.item_vector_dim,
            'item_feature_slices': self.item_feature_slices,
            'item_ablation_mode': self.ablation_mode,
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
