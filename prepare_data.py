"""
数据预处理脚本
从 MovieLens ml-latest-small 数据集生成协同过滤所需的数据文件
"""

import numpy as np
import pandas as pd
import os
import re

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'ml-latest-small')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_year(title):
    """从电影标题中提取年份"""
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return None


def prepare_movie_list():
    """
    处理 movies.csv 和 ratings.csv，生成 small_movie_list.csv
    只保留 2000 年后的电影
    """
    print("=" * 50)
    print("步骤 1: 生成电影列表 (small_movie_list.csv)")
    print("=" * 50)

    # 读取原始数据
    movies_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'movies.csv'))
    ratings_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'ratings.csv'))

    print(f"原始电影数量: {len(movies_df)}")
    print(f"原始评分数量: {len(ratings_df)}")

    # 提取年份并过滤 2000 年后的电影
    movies_df['year'] = movies_df['title'].apply(extract_year)
    movies_filtered = movies_df[movies_df['year'] >= 2000].copy()
    print(f"2000年后的电影数量: {len(movies_filtered)}")

    # 计算每部电影的平均评分和评分数量
    rating_stats = ratings_df.groupby('movieId').agg(
        mean_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

    # 合并数据
    movies_with_stats = movies_filtered.merge(rating_stats, on='movieId', how='left')

    # 只保留有评分的电影
    movies_with_stats = movies_with_stats[movies_with_stats['num_ratings'].notna()]
    movies_with_stats['mean_rating'] = movies_with_stats['mean_rating'].round(2)
    movies_with_stats['num_ratings'] = movies_with_stats['num_ratings'].astype(int)

    print(f"有评分的电影数量: {len(movies_with_stats)}")

    # 重置索引（从0开始连续编号）
    movies_with_stats = movies_with_stats.reset_index(drop=True)

    # 创建 movieId 到新索引的映射
    movie_id_to_idx = dict(zip(movies_with_stats['movieId'], movies_with_stats.index))

    # 生成输出格式
    output_df = pd.DataFrame({
        'title': movies_with_stats['title'],
        'mean rating': movies_with_stats['mean_rating'],
        'number of ratings': movies_with_stats['num_ratings']
    })

    # 保存
    output_path = os.path.join(OUTPUT_DIR, 'small_movie_list.csv')
    output_df.to_csv(output_path)
    print(f"已保存: {output_path}")

    return movies_with_stats, movie_id_to_idx


def prepare_ratings_matrix(movies_with_stats, movie_id_to_idx):
    """
    生成评分矩阵 Y 和指示矩阵 R，保存为 small_ratings.npz
    """
    print("\n" + "=" * 50)
    print("步骤 2: 生成评分矩阵 (small_ratings.npz)")
    print("=" * 50)

    ratings_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'ratings.csv'))

    # 只保留在过滤后电影列表中的评分
    valid_movie_ids = set(movie_id_to_idx.keys())
    ratings_filtered = ratings_df[ratings_df['movieId'].isin(valid_movie_ids)].copy()

    # 创建用户ID到索引的映射
    unique_users = ratings_filtered['userId'].unique()
    user_id_to_idx = {uid: idx for idx, uid in enumerate(sorted(unique_users))}

    num_movies = len(movie_id_to_idx)
    num_users = len(user_id_to_idx)

    print(f"电影数量: {num_movies}")
    print(f"用户数量: {num_users}")
    print(f"评分数量: {len(ratings_filtered)}")

    # 初始化矩阵
    Y = np.zeros((num_movies, num_users))
    R = np.zeros((num_movies, num_users))

    # 填充矩阵
    for _, row in ratings_filtered.iterrows():
        movie_idx = movie_id_to_idx[row['movieId']]
        user_idx = user_id_to_idx[row['userId']]
        Y[movie_idx, user_idx] = row['rating']
        R[movie_idx, user_idx] = 1

    # 保存
    output_path = os.path.join(OUTPUT_DIR, 'small_ratings.npz')
    np.savez(output_path, Y=Y, R=R)
    print(f"已保存: {output_path}")
    print(f"Y 矩阵形状: {Y.shape}")
    print(f"R 矩阵形状: {R.shape}")
    print(f"评分稀疏度: {R.sum() / R.size * 100:.2f}%")

    return num_movies, num_users


def prepare_initial_params(num_movies, num_users, num_features=10):
    """
    生成初始参数 X, W, b，保存为 small_params.npz
    """
    print("\n" + "=" * 50)
    print("步骤 3: 生成初始参数 (small_params.npz)")
    print("=" * 50)

    np.random.seed(42)

    X = np.random.randn(num_movies, num_features) * 0.5
    W = np.random.randn(num_users, num_features) * 0.5
    b = np.random.randn(1, num_users) * 0.1

    # 保存
    output_path = os.path.join(OUTPUT_DIR, 'small_params.npz')
    np.savez(output_path, X=X, W=W, b=b)
    print(f"已保存: {output_path}")
    print(f"X 矩阵形状: {X.shape}")
    print(f"W 矩阵形状: {W.shape}")
    print(f"b 矩阵形状: {b.shape}")


def main():
    print("开始处理 MovieLens 数据集...\n")

    # 检查原始数据是否存在
    required_files = ['movies.csv', 'ratings.csv']
    for f in required_files:
        path = os.path.join(RAW_DATA_DIR, f)
        if not os.path.exists(path):
            print(f"错误: 找不到文件 {path}")
            return

    # 执行数据处理
    movies_with_stats, movie_id_to_idx = prepare_movie_list()
    num_movies, num_users = prepare_ratings_matrix(movies_with_stats, movie_id_to_idx)
    prepare_initial_params(num_movies, num_users)

    print("\n" + "=" * 50)
    print("数据处理完成!")
    print("=" * 50)
    print(f"生成的文件位于: {OUTPUT_DIR}")
    print("  - small_movie_list.csv")
    print("  - small_ratings.npz")
    print("  - small_params.npz")


if __name__ == '__main__':
    main()
