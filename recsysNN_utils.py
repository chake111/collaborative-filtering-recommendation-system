"""
recsysNN_utils.py
基于深度学习的内容过滤推荐系统工具函数
"""

import numpy as np
import pandas as pd
import os
from collections import defaultdict
import tabulate

# 定义类型列表（14种）
GENRES = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance',
          'Sci-Fi', 'Thriller']

# 用户特征名
USER_FEATURES = ['user_id', 'rating_count', 'rating_ave'] + GENRES

# 项目/电影特征名
ITEM_FEATURES = ['movie_id', 'year', 'ave_rating'] + GENRES


def load_data():
    """
    加载并准备训练数据

    返回:
        item_train (ndarray): 项目训练数据 (num_samples, num_item_features)
        user_train (ndarray): 用户训练数据 (num_samples, num_user_features)
        y_train (ndarray): 评分标签 (num_samples,)
        item_features (list): 项目特征名列表
        user_features (list): 用户特征名列表
        item_vecs (ndarray): 所有电影的特征向量
        movie_dict (dict): 电影ID到电影信息的映射
        user_to_genre (dict): 用户到类型偏好的映射
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_dir, 'ml-latest-small', 'movies.csv')

    # 加载电影数据
    if os.path.exists(movies_path):
        movies_df = pd.read_csv(movies_path)
    else:
        # 创建合成电影数据
        movies_df = _create_synthetic_movies()

    # 设置随机种子以保证可重复性
    np.random.seed(42)

    # 处理电影数据，提取年份和类型
    movie_dict = {}
    item_vecs_list = []

    for _, row in movies_df.iterrows():
        movie_id = row['movieId']
        title = row['title']
        genres_str = row['genres'] if pd.notna(row['genres']) else ''

        # 提取年份
        year = _extract_year(title)

        # 解析类型
        genre_vec = _parse_genres(genres_str)

        # 生成平均评分（合成数据）
        ave_rating = np.round(np.random.uniform(2.5, 4.5), 1)

        # 存储电影信息
        movie_dict[movie_id] = {
            'title': title,
            'genres': genres_str,
            'year': year
        }

        # 创建项目向量: [movie_id, year, ave_rating, ...genre_one_hot...]
        item_vec = [movie_id, year, ave_rating] + genre_vec
        item_vecs_list.append(item_vec)

    item_vecs = np.array(item_vecs_list)

    # 筛选2000年以后的电影
    mask = item_vecs[:, 1] >= 2000
    item_vecs = item_vecs[mask]

    # 限制电影数量
    if len(item_vecs) > 694:
        item_vecs = item_vecs[:694]

    # 更新 movie_dict 只保留筛选后的电影
    valid_movie_ids = set(item_vecs[:, 0].astype(int))
    movie_dict = {k: v for k, v in movie_dict.items() if k in valid_movie_ids}

    # 生成用户数据
    num_users = 395
    user_to_genre = {}

    users_data = []
    for uid in range(1, num_users + 1):
        rating_count = np.random.randint(10, 100)
        rating_ave = np.round(np.random.uniform(2.0, 4.5), 1)

        # 为每个类型生成用户偏好评分
        genre_ratings = [np.round(np.random.uniform(1.5, 4.5), 1) for _ in GENRES]

        user_vec = [uid, rating_count, rating_ave] + genre_ratings
        users_data.append(user_vec)

        user_to_genre[uid] = dict(zip(GENRES, genre_ratings))

    users_array = np.array(users_data)

    # 生成训练样本（用户-电影-评分三元组）
    item_train_list = []
    user_train_list = []
    y_train_list = []

    num_movies = len(item_vecs)

    for user_idx, user_vec in enumerate(users_array):
        uid = int(user_vec[0])
        # 每个用户评价一些电影
        num_ratings = np.random.randint(20, 60)
        rated_movies = np.random.choice(num_movies, size=min(num_ratings, num_movies), replace=False)

        for movie_idx in rated_movies:
            item_vec = item_vecs[movie_idx]

            # 生成评分（基于用户类型偏好和电影类型）
            rating = _generate_rating(user_vec, item_vec)

            item_train_list.append(item_vec)
            user_train_list.append(user_vec)
            y_train_list.append(rating)

    item_train = np.array(item_train_list)
    user_train = np.array(user_train_list)
    y_train = np.array(y_train_list)

    print(f"Number of training vectors: {len(item_train)}")

    return (item_train, user_train, y_train,
            ITEM_FEATURES, USER_FEATURES,
            item_vecs, movie_dict, user_to_genre)


def _extract_year(title):
    """从电影标题中提取年份"""
    import re
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return 2000  # 默认年份


def _parse_genres(genres_str):
    """将类型字符串解析为one-hot向量"""
    genre_vec = [0] * len(GENRES)
    if genres_str and genres_str != '(no genres listed)':
        for genre in genres_str.split('|'):
            genre = genre.strip()
            if genre in GENRES:
                idx = GENRES.index(genre)
                genre_vec[idx] = 1
    return genre_vec


def _generate_rating(user_vec, item_vec):
    """基于用户偏好和电影类型生成评分"""
    # 用户类型偏好从索引3开始
    user_genre_ratings = user_vec[3:3+len(GENRES)]
    # 电影类型从索引3开始
    item_genres = item_vec[3:3+len(GENRES)]

    # 计算加权评分
    if np.sum(item_genres) > 0:
        weighted_sum = np.sum(user_genre_ratings * item_genres)
        rating = weighted_sum / np.sum(item_genres)
    else:
        rating = user_vec[2]  # 使用用户平均评分

    # 添加噪声并限制范围
    rating = rating + np.random.normal(0, 0.3)
    rating = np.clip(rating, 0.5, 5.0)
    rating = np.round(rating * 2) / 2  # 四舍五入到0.5

    return rating


def _create_synthetic_movies():
    """创建合成电影数据"""
    movies_data = []
    for i in range(1, 1000):
        year = np.random.randint(1995, 2020)
        # 随机选择1-3个类型
        num_genres = np.random.randint(1, 4)
        selected_genres = np.random.choice(GENRES, size=num_genres, replace=False)
        genres_str = '|'.join(selected_genres)

        title = f"Movie {i} ({year})"
        movies_data.append({
            'movieId': i,
            'title': title,
            'genres': genres_str
        })

    return pd.DataFrame(movies_data)


def pprint_train(data, features, vstart, start, maxcount=5, user=True):
    """
    打印训练数据

    参数:
        data: 训练数据数组
        features: 特征名列表
        vstart: 类型向量起始索引
        start: 训练使用的起始索引
        maxcount: 显示行数
        user: 是否为用户数据
    """
    # 构建表头
    headers = []
    for i, f in enumerate(features):
        if i < start:
            headers.append(f"[{f}]")  # 不用于训练的特征
        else:
            headers.append(f)

    # 构建数据行
    rows = []
    for i in range(min(maxcount, len(data))):
        row = []
        for j, val in enumerate(data[i]):
            if j < vstart:
                row.append(f"{val:.1f}" if isinstance(val, float) else str(int(val)))
            else:
                # 类型特征
                if abs(val) < 0.01:
                    row.append("")
                else:
                    row.append(f"{val:.2f}")
        rows.append(row)

    table = tabulate.tabulate(rows, headers=headers, tablefmt='pretty')
    print(table)


def gen_user_vecs(user_vec, num_items):
    """
    生成重复的用户向量以匹配电影数量

    参数:
        user_vec: 单个用户向量 (1, num_features)
        num_items: 电影数量

    返回:
        user_vecs: 重复的用户向量 (num_items, num_features)
    """
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs


def predict_uservec(user_vecs, item_vecs, model, u_s, i_s, scaler,
                    scalerUser, scalerItem, scaledata=True):
    """
    使用模型预测用户对所有电影的评分

    参数:
        user_vecs: 用户向量
        item_vecs: 电影向量
        model: 训练好的模型
        u_s: 用户特征起始索引
        i_s: 项目特征起始索引
        scaler: 目标缩放器
        scalerUser: 用户特征缩放器
        scalerItem: 项目特征缩放器
        scaledata: 是否缩放数据

    返回:
        sorted_index: 按预测评分排序的索引
        sorted_ypu: 排序后的预测评分
        sorted_items: 排序后的项目向量
        sorted_user: 排序后的用户向量
    """
    if scaledata:
        scaled_user_vecs = scalerUser.transform(user_vecs)
        scaled_item_vecs = scalerItem.transform(item_vecs)
    else:
        scaled_user_vecs = user_vecs
        scaled_item_vecs = item_vecs

    # 预测
    y_pred = model.predict([scaled_user_vecs[:, u_s:], scaled_item_vecs[:, i_s:]])

    # 反变换到原始评分范围
    y_pred_unscaled = scaler.inverse_transform(y_pred)

    # 按评分降序排序
    sorted_index = np.argsort(-y_pred_unscaled.flatten())
    sorted_ypu = y_pred_unscaled[sorted_index]
    sorted_items = item_vecs[sorted_index]
    sorted_user = user_vecs[sorted_index]

    return sorted_index, sorted_ypu, sorted_items, sorted_user


def print_pred_movies(sorted_ypu, sorted_user, sorted_items, movie_dict, maxcount=10):
    """
    打印预测的电影推荐

    参数:
        sorted_ypu: 排序后的预测评分
        sorted_user: 排序后的用户向量
        sorted_items: 排序后的项目向量
        movie_dict: 电影信息字典
        maxcount: 显示数量
    """
    print(f"\n{'='*80}")
    print(f"{'Top Movie Recommendations':^80}")
    print(f"{'='*80}")

    headers = ["Rank", "Pred Rating", "Year", "Movie Title", "Genres"]
    rows = []

    for i in range(min(maxcount, len(sorted_ypu))):
        movie_id = int(sorted_items[i, 0])
        year = int(sorted_items[i, 1])
        pred_rating = sorted_ypu[i, 0] if sorted_ypu.ndim > 1 else sorted_ypu[i]

        if movie_id in movie_dict:
            title = movie_dict[movie_id]['title']
            genres = movie_dict[movie_id]['genres']
        else:
            title = f"Movie {movie_id}"
            genres = "Unknown"

        rows.append([i+1, f"{pred_rating:.2f}", year, title, genres])

    table = tabulate.tabulate(rows, headers=headers, tablefmt='pretty')
    print(table)


def get_user_vecs(uid, user_train, item_vecs, user_to_genre):
    """
    获取指定用户的向量

    参数:
        uid: 用户ID
        user_train: 用户训练数据
        item_vecs: 项目向量
        user_to_genre: 用户类型偏好映射

    返回:
        user_vecs: 用户向量（重复以匹配电影数量）
        y_vecs: 对应的真实评分（如果有）
    """
    # 找到该用户的数据
    user_mask = user_train[:, 0] == uid

    if np.any(user_mask):
        user_vec = user_train[user_mask][0:1]  # 取第一条用户记录
    else:
        # 如果找不到，创建一个默认用户向量
        default_genre_ratings = [3.0] * len(GENRES)
        user_vec = np.array([[uid, 50, 3.5] + default_genre_ratings])

    num_items = len(item_vecs)
    user_vecs = np.tile(user_vec, (num_items, 1))

    # 生成模拟的真实评分
    y_vecs = np.random.uniform(2.0, 4.5, size=num_items)

    return user_vecs, y_vecs


def print_existing_user(sorted_ypu, sorted_y, sorted_user, sorted_items,
                        item_features, ivs, uvs, movie_dict, maxcount=10):
    """
    打印现有用户的预测结果与真实评分对比

    参数:
        sorted_ypu: 排序后的预测评分
        sorted_y: 排序后的真实评分
        sorted_user: 排序后的用户向量
        sorted_items: 排序后的项目向量
        item_features: 项目特征名
        ivs: 项目类型向量起始索引
        uvs: 用户类型向量起始索引
        movie_dict: 电影信息字典
        maxcount: 显示数量
    """
    print(f"\n{'='*90}")
    print(f"{'Predictions for Existing User':^90}")
    print(f"{'='*90}")

    headers = ["Rank", "Pred", "Actual", "Diff", "Year", "Movie Title", "Genres"]
    rows = []

    for i in range(min(maxcount, len(sorted_ypu))):
        movie_id = int(sorted_items[i, 0])
        year = int(sorted_items[i, 1])
        pred_rating = sorted_ypu[i, 0] if sorted_ypu.ndim > 1 else sorted_ypu[i]
        actual_rating = sorted_y[i, 0] if sorted_y.ndim > 1 else sorted_y[i]
        diff = pred_rating - actual_rating

        if movie_id in movie_dict:
            title = movie_dict[movie_id]['title']
            genres = movie_dict[movie_id]['genres']
        else:
            title = f"Movie {movie_id}"
            genres = "Unknown"

        # 截断标题
        if len(title) > 35:
            title = title[:32] + "..."

        rows.append([i+1, f"{pred_rating:.2f}", f"{actual_rating:.2f}",
                     f"{diff:+.2f}", year, title, genres[:30]])

    table = tabulate.tabulate(rows, headers=headers, tablefmt='pretty')
    print(table)


def get_item_genre(item_vec, ivs, item_features):
    """
    从项目向量获取类型信息

    参数:
        item_vec: 项目向量
        ivs: 类型向量起始索引
        item_features: 项目特征名

    返回:
        genre_str: 类型字符串
        genre_list: 类型列表
    """
    genre_list = []
    genre_indices = item_vec[ivs:]

    for i, val in enumerate(genre_indices):
        if val > 0.5 and i < len(GENRES):
            genre_list.append(GENRES[i])

    genre_str = '|'.join(genre_list) if genre_list else 'Unknown'

    return genre_str, genre_list


# 导出所有函数
__all__ = [
    'load_data',
    'pprint_train',
    'gen_user_vecs',
    'predict_uservec',
    'print_pred_movies',
    'get_user_vecs',
    'print_existing_user',
    'get_item_genre',
    'GENRES',
    'USER_FEATURES',
    'ITEM_FEATURES'
]
