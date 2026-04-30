import json
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AmazonFashionAnalyzer:
    def __init__(self, review_path, meta_path):
        self.review_path = review_path
        self.meta_path = meta_path
        self.reviews = []
        self.meta_data = []
        self.review_df = None
        self.meta_df = None
        
    def load_data(self):
        print("=" * 60)
        print("开始加载数据...")
        print("=" * 60)
        
        print("\n1. 加载评论文本数据...")
        with open(self.review_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.reviews.append(json.loads(line))
        print(f"   评论文本数据加载完成，共 {len(self.reviews)} 条记录")
        
        print("\n2. 加载商品元数据...")
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        self.meta_data.append(json.loads(line))
                    except:
                        continue
        print(f"   商品元数据加载完成，共 {len(self.meta_data)} 条记录")
        
        self.review_df = pd.DataFrame(self.reviews)
        self.meta_df = pd.DataFrame(self.meta_data)
        
    def analyze_review_structure(self):
        print("\n" + "=" * 60)
        print("一、FASHION子集结构分析")
        print("=" * 60)
        
        print("\n(一) 评论文本数据结构")
        print("-" * 60)
        print(f"1. 数据集基本信息:")
        print(f"   - 总评论数: {len(self.review_df):,}")
        print(f"   - 数据列数: {len(self.review_df.columns)}")
        print(f"   - 数据列: {list(self.review_df.columns)}")
        
        print(f"\n2. 数据类型分析:")
        print(self.review_df.dtypes)
        
        print(f"\n3. 缺失值统计:")
        missing_stats = self.review_df.isnull().sum()
        for col, count in missing_stats.items():
            if count > 0:
                print(f"   - {col}: {count} ({count/len(self.review_df)*100:.2f}%)")
        
        print(f"\n4. 数据样例:")
        print(self.review_df.head(2).to_string())
        
    def analyze_user_statistics(self):
        print("\n(二) 用户统计信息")
        print("-" * 60)
        
        user_stats = self.review_df.groupby('reviewerID').agg({
            'asin': 'count',
            'overall': ['mean', 'std'],
            'unixReviewTime': ['min', 'max']
        }).reset_index()
        user_stats.columns = ['user_id', 'review_count', 'avg_rating', 'rating_std', 'first_review', 'last_review']
        
        print(f"1. 用户数量: {len(user_stats):,}")
        print(f"2. 用户评论数分布:")
        print(f"   - 平均评论数: {user_stats['review_count'].mean():.2f}")
        print(f"   - 中位数: {user_stats['review_count'].median():.2f}")
        print(f"   - 最大值: {user_stats['review_count'].max()}")
        print(f"   - 最小值: {user_stats['review_count'].min()}")
        
        print(f"\n3. 用户活跃度分布:")
        print(f"   - 评论数 > 5: {(user_stats['review_count'] > 5).sum()} ({(user_stats['review_count'] > 5).sum()/len(user_stats)*100:.2f}%)")
        print(f"   - 评论数 > 10: {(user_stats['review_count'] > 10).sum()} ({(user_stats['review_count'] > 10).sum()/len(user_stats)*100:.2f}%)")
        print(f"   - 评论数 > 20: {(user_stats['review_count'] > 20).sum()} ({(user_stats['review_count'] > 20).sum()/len(user_stats)*100:.2f}%)")
        
        print(f"\n4. 用户评分分布:")
        print(f"   - 平均评分: {user_stats['avg_rating'].mean():.2f}")
        print(f"   - 评分标准差: {user_stats['avg_rating'].std():.2f}")
        
        return user_stats
        
    def analyze_item_statistics(self):
        print("\n(三) 商品统计信息")
        print("-" * 60)
        
        item_stats = self.review_df.groupby('asin').agg({
            'reviewerID': 'count',
            'overall': ['mean', 'std']
        }).reset_index()
        item_stats.columns = ['item_id', 'review_count', 'avg_rating', 'rating_std']
        
        print(f"1. 商品数量: {len(item_stats):,}")
        print(f"2. 商品评论数分布:")
        print(f"   - 平均评论数: {item_stats['review_count'].mean():.2f}")
        print(f"   - 中位数: {item_stats['review_count'].median():.2f}")
        print(f"   - 最大值: {item_stats['review_count'].max()}")
        print(f"   - 最小值: {item_stats['review_count'].min()}")
        
        print(f"\n3. 商品热度分布:")
        print(f"   - 评论数 > 10: {(item_stats['review_count'] > 10).sum()} ({(item_stats['review_count'] > 10).sum()/len(item_stats)*100:.2f}%)")
        print(f"   - 评论数 > 50: {(item_stats['review_count'] > 50).sum()} ({(item_stats['review_count'] > 50).sum()/len(item_stats)*100:.2f}%)")
        print(f"   - 评论数 > 100: {(item_stats['review_count'] > 100).sum()} ({(item_stats['review_count'] > 100).sum()/len(item_stats)*100:.2f}%)")
        
        print(f"\n4. 商品评分分布:")
        print(f"   - 平均评分: {item_stats['avg_rating'].mean():.2f}")
        print(f"   - 评分标准差: {item_stats['avg_rating'].std():.2f}")
        
        return item_stats
        
    def analyze_rating_distribution(self):
        print("\n(四) 评分分布分析")
        print("-" * 60)
        
        rating_counts = self.review_df['overall'].value_counts().sort_index()
        print(f"1. 评分统计:")
        for rating, count in rating_counts.items():
            print(f"   - {rating}星: {count:,} ({count/len(self.review_df)*100:.2f}%)")
        
        print(f"\n2. 评分统计量:")
        print(f"   - 平均评分: {self.review_df['overall'].mean():.2f}")
        print(f"   - 中位数: {self.review_df['overall'].median():.2f}")
        print(f"   - 标准差: {self.review_df['overall'].std():.2f}")
        
        print(f"\n3. 验证购买统计:")
        if 'verified' in self.review_df.columns:
            verified_counts = self.review_df['verified'].value_counts()
            print(f"   - 验证购买: {verified_counts.get(True, 0):,} ({verified_counts.get(True, 0)/len(self.review_df)*100:.2f}%)")
            print(f"   - 未验证: {verified_counts.get(False, 0):,} ({verified_counts.get(False, 0)/len(self.review_df)*100:.2f}%)")
        
    def analyze_temporal_distribution(self):
        print("\n(五) 时间分布分析")
        print("-" * 60)
        
        self.review_df['review_date'] = pd.to_datetime(self.review_df['unixReviewTime'], unit='s')
        
        print(f"1. 时间范围:")
        print(f"   - 最早评论: {self.review_df['review_date'].min()}")
        print(f"   - 最晚评论: {self.review_df['review_date'].max()}")
        print(f"   - 时间跨度: {(self.review_df['review_date'].max() - self.review_df['review_date'].min()).days} 天")
        
        self.review_df['review_year'] = self.review_df['review_date'].dt.year
        yearly_counts = self.review_df['review_year'].value_counts().sort_index()
        print(f"\n2. 年度评论分布:")
        for year, count in yearly_counts.items():
            print(f"   - {year}年: {count:,} ({count/len(self.review_df)*100:.2f}%)")
        
    def analyze_meta_structure(self):
        print("\n" + "=" * 60)
        print("二、元数据与评论文本分析")
        print("=" * 60)
        
        print("\n(一) 商品元数据结构")
        print("-" * 60)
        print(f"1. 元数据基本信息:")
        print(f"   - 总商品数: {len(self.meta_df):,}")
        print(f"   - 数据列数: {len(self.meta_df.columns)}")
        print(f"   - 数据列: {list(self.meta_df.columns)}")
        
        print(f"\n2. 数据类型分析:")
        print(self.meta_df.dtypes)
        
        print(f"\n3. 缺失值统计:")
        missing_stats = self.meta_df.isnull().sum()
        for col, count in missing_stats.items():
            if count > 0:
                print(f"   - {col}: {count} ({count/len(self.meta_df)*100:.2f}%)")
        
        print(f"\n4. 数据样例:")
        print(self.meta_df.head(2).to_string())
        
    def analyze_product_features(self):
        print("\n(二) 商品特征分析")
        print("-" * 60)
        
        if 'brand' in self.meta_df.columns:
            brand_counts = self.meta_df['brand'].value_counts().head(20)
            print(f"1. 品牌分布 (Top 20):")
            for brand, count in brand_counts.items():
                print(f"   - {brand}: {count}")
            print(f"   - 总品牌数: {self.meta_df['brand'].nunique():,}")
        
        if 'imageURL' in self.meta_df.columns:
            image_stats = self.meta_df['imageURL'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            print(f"\n2. 图片信息:")
            print(f"   - 有图片的商品: {(image_stats > 0).sum()} ({(image_stats > 0).sum()/len(self.meta_df)*100:.2f}%)")
            print(f"   - 平均图片数: {image_stats.mean():.2f}")
            print(f"   - 最大图片数: {image_stats.max()}")
        
        if 'category' in self.meta_df.columns:
            print(f"\n3. 类别信息:")
            print(f"   - 有类别信息的商品: {self.meta_df['category'].notna().sum()} ({self.meta_df['category'].notna().sum()/len(self.meta_df)*100:.2f}%)")
        
        if 'price' in self.meta_df.columns:
            print(f"\n4. 价格信息:")
            print(f"   - 有价格信息的商品: {self.meta_df['price'].notna().sum()} ({self.meta_df['price'].notna().sum()/len(self.meta_df)*100:.2f}%)")
            
    def analyze_review_text_features(self):
        print("\n(三) 评论文本特征分析")
        print("-" * 60)
        
        if 'reviewText' in self.review_df.columns:
            text_lengths = self.review_df['reviewText'].apply(lambda x: len(str(x)))
            print(f"1. 评论文本长度:")
            print(f"   - 平均长度: {text_lengths.mean():.2f} 字符")
            print(f"   - 中位数: {text_lengths.median():.2f} 字符")
            print(f"   - 最大值: {text_lengths.max()} 字符")
            print(f"   - 最小值: {text_lengths.min()} 字符")
        
        if 'summary' in self.review_df.columns:
            summary_lengths = self.review_df['summary'].apply(lambda x: len(str(x)))
            print(f"\n2. 评论摘要长度:")
            print(f"   - 平均长度: {summary_lengths.mean():.2f} 字符")
            print(f"   - 中位数: {summary_lengths.median():.2f} 字符")
            print(f"   - 最大值: {summary_lengths.max()} 字符")
        
        if 'vote' in self.review_df.columns:
            print(f"\n3. 评论投票信息:")
            print(f"   - 有投票信息的评论: {self.review_df['vote'].notna().sum()} ({self.review_df['vote'].notna().sum()/len(self.review_df)*100:.2f}%)")
            
    def analyze_data_sparsity(self):
        print("\n(四) 数据稀疏性分析")
        print("-" * 60)
        
        n_users = self.review_df['reviewerID'].nunique()
        n_items = self.review_df['asin'].nunique()
        n_reviews = len(self.review_df)
        
        print(f"1. 用户-商品矩阵信息:")
        print(f"   - 用户数: {n_users:,}")
        print(f"   - 商品数: {n_items:,}")
        print(f"   - 评论数: {n_reviews:,}")
        print(f"   - 矩阵密度: {n_reviews/(n_users*n_items)*100:.4f}%")
        print(f"   - 矩阵稀疏度: {(1-n_reviews/(n_users*n_items))*100:.4f}%")
        
        print(f"\n2. 长尾分布分析:")
        user_review_counts = self.review_df['reviewerID'].value_counts()
        item_review_counts = self.review_df['asin'].value_counts()
        
        print(f"   - 用户长尾: Top 20%用户贡献 {(user_review_counts[:int(len(user_review_counts)*0.2)].sum()/n_reviews*100):.2f}% 评论")
        print(f"   - 商品长尾: Top 20%商品贡献 {(item_review_counts[:int(len(item_review_counts)*0.2)].sum()/n_reviews*100):.2f}% 评论")
        
    def generate_summary_report(self):
        print("\n" + "=" * 60)
        print("三、数据集分析总结")
        print("=" * 60)
        
        print("\n1. 数据集规模:")
        print(f"   - 评论总数: {len(self.review_df):,}")
        print(f"   - 用户总数: {self.review_df['reviewerID'].nunique():,}")
        print(f"   - 商品总数: {self.review_df['asin'].nunique():,}")
        print(f"   - 元数据商品数: {len(self.meta_df):,}")
        
        print(f"\n2. 数据质量:")
        print(f"   - 平均用户评论数: {self.review_df.groupby('reviewerID').size().mean():.2f}")
        print(f"   - 平均商品评论数: {self.review_df.groupby('asin').size().mean():.2f}")
        print(f"   - 矩阵稀疏度: {(1-len(self.review_df)/(self.review_df['reviewerID'].nunique()*self.review_df['asin'].nunique()))*100:.2f}%")
        
        print(f"\n3. 特征丰富度:")
        print(f"   - 评论文本覆盖率: {(self.review_df['reviewText'].notna().sum()/len(self.review_df)*100):.2f}%")
        print(f"   - 图片覆盖率: {(self.meta_df['imageURL'].apply(lambda x: len(x) if isinstance(x, list) else 0) > 0).sum()/len(self.meta_df)*100:.2f}%")
        print(f"   - 品牌信息覆盖率: {(self.meta_df['brand'].notna().sum()/len(self.meta_df)*100):.2f}%")
        
        print(f"\n4. 数据特点:")
        print(f"   - 数据稀疏性高，符合推荐系统特点")
        print(f"   - 存在明显长尾分布")
        print(f"   - 多模态特征丰富(文本、图像、元数据)")
        print(f"   - 时间跨度大，适合时序分析")
        
    def run_full_analysis(self):
        self.load_data()
        self.analyze_review_structure()
        self.analyze_user_statistics()
        self.analyze_item_statistics()
        self.analyze_rating_distribution()
        self.analyze_temporal_distribution()
        self.analyze_meta_structure()
        self.analyze_product_features()
        self.analyze_review_text_features()
        self.analyze_data_sparsity()
        self.generate_summary_report()
        
        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)

if __name__ == "__main__":
    analyzer = AmazonFashionAnalyzer(
        review_path="dataset/AMAZON_FASHION.json",
        meta_path="dataset/meta_AMAZON_FASHION.json"
    )
    analyzer.run_full_analysis()