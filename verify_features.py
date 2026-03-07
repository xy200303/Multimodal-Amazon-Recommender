import pandas as pd
import numpy as np

print("Loading feature data...")
feat_df = pd.read_csv('new_feat/item.csv')

print(f"\n=== 数据集基本信息 ===")
print(f"总商品数: {len(feat_df)}")
print(f"总特征数: {len(feat_df.columns)}")

print(f"\n=== 特征分类 ===")
original_cols = ['asin', 'title', 'brand', 'feature', 'description', 'price', 'imageURL', 
                 'imageURLHighRes', 'rank', 'rank_num', 'rank_field', 'date', 'also_view', 
                 'also_buy', 'fit', 'details', 'similar_item', 'tech1']
new_cols = [col for col in feat_df.columns if col not in original_cols]

title_clip_col = 'title_clip' if 'title_clip' in new_cols else None
image_clip_col = 'image_clip' if 'image_clip' in new_cols else None
keyword_col = 'title_keywords' if 'title_keywords' in new_cols else None

print(f"原始字段数: {len(original_cols)}")
print(f"标题CLIP向量字段: {'存在' if title_clip_col else '不存在'}")
print(f"图像CLIP向量字段: {'存在' if image_clip_col else '不存在'}")
print(f"关键词字段: {'存在' if keyword_col else '不存在'}")
print(f"其他新增字段: {len(new_cols) - (1 if title_clip_col else 0) - (1 if image_clip_col else 0) - (1 if keyword_col else 0)}")

print(f"\n=== 缺失值统计 ===")
missing_stats = feat_df.isnull().sum()
missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)

if len(missing_stats) > 0:
    print(f"有缺失值的字段数: {len(missing_stats)}")
    print(f"缺失值最多的10个字段:")
    for col, count in missing_stats.head(10).items():
        print(f"  {col}: {count} ({count/len(feat_df)*100:.2f}%)")
else:
    print("✓ 所有字段都没有缺失值")

print(f"\n=== CLIP向量统计 ===")
if title_clip_col:
    import ast
    title_vectors = np.array([np.array(ast.literal_eval(v)) if isinstance(v, str) else np.array(v) for v in feat_df[title_clip_col]])
    print(f"标题CLIP向量:")
    print(f"  形状: {title_vectors.shape}")
    print(f"  均值: {title_vectors.mean():.6f}")
    print(f"  标准差: {title_vectors.std():.6f}")
    print(f"  最小值: {title_vectors.min():.6f}")
    print(f"  最大值: {title_vectors.max():.6f}")

if image_clip_col:
    import ast
    image_vectors = np.array([np.array(ast.literal_eval(v)) if isinstance(v, str) else np.array(v) for v in feat_df[image_clip_col]])
    print(f"\n图像CLIP向量:")
    print(f"  形状: {image_vectors.shape}")
    print(f"  均值: {image_vectors.mean():.6f}")
    print(f"  标准差: {image_vectors.std():.6f}")
    print(f"  最小值: {image_vectors.min():.6f}")
    print(f"  最大值: {image_vectors.max():.6f}")

print(f"\n=== 关键词统计 ===")
if keyword_col:
    keywords = feat_df[keyword_col]
    non_empty_keywords = keywords[keywords != '']
    print(f"有关键词的商品数: {len(non_empty_keywords)}")
    print(f"关键词为空的商品数: {len(keywords) - len(non_empty_keywords)}")
    
    if len(non_empty_keywords) > 0:
        avg_keyword_length = non_empty_keywords.str.split().str.len().mean()
        print(f"平均关键词数量: {avg_keyword_length:.1f}")
        
        print(f"\n关键词示例 (前5个):")
        for i, kw in enumerate(non_empty_keywords.head(5)):
            print(f"  {i+1}. {kw}")

print(f"\n=== 数据类型统计 ===")
dtype_counts = feat_df.dtypes.value_counts()
print(f"数据类型分布:")
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} 个字段")

print(f"\n=== 示例数据 (前3个商品) ===")
sample_cols = ['asin', 'title', 'brand', 'title_keywords']
if title_clip_col:
    sample_cols.append(title_clip_col)
if image_clip_col:
    sample_cols.append(image_clip_col)

print(f"显示的字段: {sample_cols}")
sample_df = feat_df[sample_cols].head(3).copy()

if title_clip_col:
    sample_df[title_clip_col] = sample_df[title_clip_col].apply(lambda x: f"[{len(x)}维向量]" if isinstance(x, list) else str(x))
if image_clip_col:
    sample_df[image_clip_col] = sample_df[image_clip_col].apply(lambda x: f"[{len(x)}维向量]" if isinstance(x, list) else str(x))

print(sample_df.to_string())

print(f"\n=== 验证完成 ===")
print("✓ 特征数据集构建成功!")
