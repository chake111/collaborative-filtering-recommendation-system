import numpy as np
import pandas as pd
import os

# ===================== 数据加载函数 =====================

def load_precalc_params_small():
    """
    加载预计算的协同过滤参数 X, W, b
    返回:
        X (ndarray): (num_movies, num_features) 电影特征矩阵
        W (ndarray): (num_users, num_features)  用户参数矩阵
        b (ndarray): (1, num_users)              用户偏置向量
        num_movies  (int): 电影数量
        num_features(int): 特征数量
        num_users   (int): 用户数量
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'small_params.npz')

    if os.path.exists(file_path):
        data = np.load(file_path)
        X = data['X']
        W = data['W']
        b = data['b']
    else:
        # 使用固定随机种子生成合成数据
        np.random.seed(42)
        num_movies   = 4778
        num_users    = 443
        num_features = 10
        X = np.random.randn(num_movies,   num_features) * 0.5
        W = np.random.randn(num_users,    num_features) * 0.5
        b = np.random.randn(1,            num_users)    * 0.1

    num_movies, num_features = X.shape
    num_users = W.shape[0]
    return X, W, b, num_movies, num_features, num_users


def load_ratings_small():
    """
    加载电影评分数据集
    返回:
        Y (ndarray): (num_movies, num_users) 评分矩阵，0.5~5分
        R (ndarray): (num_movies, num_users) 指示矩阵，1表示已评分
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'small_ratings.npz')

    if os.path.exists(file_path):
        data = np.load(file_path)
        Y = data['Y']
        R = data['R']
    else:
        np.random.seed(42)
        num_movies = 4778
        num_users  = 443
        # 约25%的用户对电影进行了评分
        R = (np.random.rand(num_movies, num_users) > 0.75).astype(float)
        # 评分范围 0.5 ~ 5.0，步长 0.5
        raw = np.random.randint(1, 11, size=(num_movies, num_users)) * 0.5
        Y = R * raw

    return Y, R


def load_Movie_List_pd():
    """
    加载电影列表
    返回:
        movieList    (list):      电影标题列表
        movieList_df (DataFrame): 包含 title / mean rating / number of ratings 列
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'small_movie_list.csv')

    if os.path.exists(file_path):
        movieList_df = pd.read_csv(file_path, index_col=0)
        movieList = movieList_df['title'].tolist()
    else:
        num_movies = 4778
        np.random.seed(42)

        # 在笔记本中用到的特定索引处填入真实电影名
        titles = [f"Movie {i} ({2000 + i % 22})" for i in range(num_movies)]
        named = {
            2700: "Toy Story 3 (2010)",
            2609: "Persuasion (2007)",
            929:  "Lord of the Rings: The Return of the King, The (2003)",
            246:  "Shrek (2001)",
            2716: "Inception (2010)",
            1150: "Incredibles, The (2004)",
            382:  "Amelie (Fabuleux destin d'Amelie Poulain, Le) (2001)",
            366:  "Harry Potter and the Sorcerer's Stone (2001)",
            622:  "Harry Potter and the Chamber of Secrets (2002)",
            988:  "Eternal Sunshine of the Spotless Mind (2004)",
            2925: "Louis Theroux: Law & Disorder (2008)",
            2937: "Nothing to Declare (Rien a declarer) (2010)",
            793:  "Pirates of the Caribbean: The Curse of the Black Pearl (2003)",
        }
        for idx, name in named.items():
            titles[idx] = name

        mean_ratings    = np.round(np.random.rand(num_movies) * 3.5 + 1.5, 1)
        number_of_ratings = np.random.randint(1, 300, num_movies)

        movieList_df = pd.DataFrame({
            'title':             titles,
            'mean rating':       mean_ratings,
            'number of ratings': number_of_ratings
        })
        movieList = titles

    return movieList, movieList_df


# ===================== 工具函数 =====================

def normalizeRatings(Y, R):
    """
    对评分矩阵按电影（行）做均值归一化
    参数:
        Y (ndarray): (num_movies, num_users) 评分矩阵
        R (ndarray): (num_movies, num_users) 指示矩阵
    返回:
        Ynorm (ndarray): 归一化后的评分矩阵
        Ymean (ndarray): (num_movies, 1) 每部电影的平均评分
    """
    num_movies = Y.shape[0]
    Ymean = np.zeros((num_movies, 1))
    Ynorm = np.zeros(Y.shape)

    for i in range(num_movies):
        idx = R[i, :] == 1
        if np.sum(idx) > 0:
            Ymean[i] = np.mean(Y[i, idx])
            Ynorm[i, :] = (Y[i, :] - Ymean[i]) * R[i, :]

    return Ynorm, Ymean
