import csv
import math

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel  # 计算余弦相似度


# 返回正向索引与负向索引
def get_list_index_map(list):
    map_ = {}
    map_reverse = {}
    for i in range(len(list)):
        map_[list[i]] = i
        map_reverse[i] = list[i]
    return map_, map_reverse


# 返回 genres_list
def get_type_list():
    genres_list = []
    for item in genres_list:
        movie_types = item.split('|')
        for movie_type in movie_types:
            if movie_type not in genres_list:
                genres_list.append(movie_type)
    return genres_list


# 评分矩阵，用户 i 对电影 j 的评分。
def get_rating_matrix():
    matrix = np.zeros((len(user_map.keys()), len(movie_map.keys())))
    for row in ratings.itertuples(index=True, name='Pandas'):
        user = user_map[getattr(row, "userId")]
        movie = movie_map[getattr(row, "movieId")]
        rate = getattr(row, "rating")
        matrix[user, movie] = rate
    return matrix


# 用户相似度矩阵，是对称的
def get_user_sim_matrix(input_matrix):
    size = len(input_matrix)
    matrix = np.zeros((size, size))

    for i in range(size):
        sum_ = 0
        for j in range(len(input_matrix[i])):
            sum_ += input_matrix[i][j]
        avg = sum_ / len(input_matrix[i])

        for j in range(len(input_matrix[i])):
            input_matrix[i][j] -= avg  # 去掉该用户打分均值

    for i in range(size):
        for j in range(i + 1, size):
            sim = cosine_similarity(input_matrix[i], input_matrix[j])
            matrix[i, j] = matrix[j, i] = sim
    return matrix


def cosine_similarity(list1, list2):
    res = 0
    d1 = 0
    d2 = 0
    for index in range(len(list1)):
        val1 = list1[index]
        val2 = list2[index]
        res += val1 * val2
        d1 += val1 ** 2
        d2 += val2 ** 2
    return res / (math.sqrt(d1 * d2))


# 读取处理
ratings = pd.read_csv('../ml-latest-small/ratings.csv', index_col=None)
movies = pd.read_csv('../ml-latest-small/movies.csv', index_col=None)
tags = pd.read_csv('../ml-latest-small/tags.csv', index_col=None)
user_list = ratings['userId'].drop_duplicates().values.tolist()
movie_list = movies['movieId'].drop_duplicates().values.tolist()
genres_list = movies['genres'].values.tolist()
tag_list = tags['tag'].values.tolist()

type_list = get_type_list()
type_map, type_map_reverse = get_list_index_map(type_list)
user_map, user_map_reverse = get_list_index_map(user_list)
movie_map, movie_map_reverse = get_list_index_map(movie_list)

ratings_matrix = get_rating_matrix()
user_sim_matrix = get_user_sim_matrix(ratings_matrix)


# 用户-用户模型，电影评分预测
def get_predict(matrix, user_index, movie_index, k):
    line = matrix[user_index]
    neighbors = []
    for i in range(len(line)):
        neighbors.append([i, line[i]])
    neighbors.sort(key=lambda val: val[1], reverse=False)
    neighbors = neighbors[:k]  # 获取相似的用户，返回为[(12,0.88),(18,0.99)]

    similarity_sum = 0
    rate = [0 for _ in range(len(ratings_matrix[0]))]
    for pair in neighbors:
        neighbor_sim = pair[1]
        similarity_sum += neighbor_sim
        rate += ratings_matrix[movie_index] * neighbor_sim
    rate /= similarity_sum
    return rate


# k 的定义为最近邻的 k 个用户
# 返回值为[(电影1,0.87),(电影2,0.85)...]
def userCF_recommend(matrix, index, k, n):
    rate = []
    for movie_index in range(len(movie_list)):
        rate[movie_index] = get_predict(matrix, index, movie_index, k)  # 获取预测评分

    for i in range(len(rate)):
        if ratings_matrix[index][i] != 0:  # 若已经看过此部电影，则预测打分置为 0
            rate[i] = 0
    res = []
    for i in range(len(rate)):
        res.append([i, rate[i]])
    res.sort(key=lambda val: val[1], reverse=True)
    return res[:n]


# 返回为[(电影id，平均分，评分人数),,,]
def type_rank_map():
    map_ = {}
    for t in type_list:
        map_[t] = []
    for movie in range(len(genres_list)):
        rates = np.array(ratings_matrix)[:, movie]  # 计算该电影的用户均分
        count = 0
        rate = 0
        for r in rates:
            if r != 0:
                rate += r
                count += 1
        if count != 0:
            rate = rate / count
        types = genres_list[movie].split('|')
        for t in types:
            map_[t].append((movie, rate, count))

    for t in type_list:
        temp = map_[t]
        temp.sort(key=lambda val: (val[1], val[2]), reverse=True)
        map_[t] = temp
    return map_


type_rank_map = type_rank_map()


def movie_similarity():
    tfidf = TfidfVectorizer(stop_words="english")  # 英语内建的停用词表
    tfidf_matrix = tfidf.fit_transform(tags["genres"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    # 传入句，例如corpus =  ["I have a pen.","I have an apple."]
    # 一部电影可能有不同的分类，因此下面只能找到类型相似的电影（输入只有类型）
    return cosine_sim


def content_based_list(user_index):
    seen_movies = movie_map[user_index]  # 该用户看过的电影

    prefer_movies = ratings[ratings["userId"] == user_index].sort_values(by=["rating"]).head(5)[
        "movieId"].drop_duplicates()

    cosine_sim = movie_similarity()
    similarities = {}  # 记录格式为：{740:0.999,1224:0.999}之类的
    for movie_id in prefer_movies:
        similarity = list(
            enumerate(cosine_sim[movie_id]))  # 存放的是[(0,0),(1,0)...(3,0.168)]数据，对于每一元组两个数，第一个表示movieId，第二个表示相似度
        similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
        similarity = similarity[1: 10]  # 因为第0个similarity一定是自己
        for s in similarity:  # s形式为(740,0.998)，元组第一个数表示其他电影的idx，后一个数表示相似度
            if s[0] in similarities or s[0] in seen_movies:  # 表示该电影已经和之前的电影有过相似的记录了，就直接跳过
                continue
            else:
                similarities[s[0]] = s[1]
    recommend_ids = list(similarities.keys())
    recommend_ids.sort(key=lambda x: similarities[x], reverse=True)  # 再总体按照相似度进行排序

    return recommend_ids[1:5]  # 去除自身


# [i,j] 表示用户 i 对 j 类电影的喜爱程度
def get_user_favor_matrix():
    matrix = np.zeros((len(user_list), len(type_list)))
    for user in range((len(user_list))):
        weight = 0
        rating = ratings_matrix[user]
        for movie in range(len(rating)):
            if rating[movie] != 0:
                types = genres_list[movie].split('|')
                for t in types:
                    if t in type_map.keys():
                        matrix[user][type_map[t]] += rating[movie]
                        weight += rating[movie]
        matrix[user] /= weight
    return matrix


"""
获得基于物品的推荐
user_index: 
user_prefer_matrix: 用户偏好矩阵
type_rank: 每类电影排名 map
least: 至少有least个人评分才算有效
返回[(电影id,平均分，评分人数)]
"""


def itemCF_recommend(user_index):
    seen_movie_ids = ratings[ratings["userId"] == user_index]["movieId"]
    results = {}

    for movie_id in range(len(seen_movie_ids)):

        # 电影相似度
        similarity_movies = []
        weight_similarity = 0.0
        total_similarity = 0.0
        movie_similarity_matrix = movie_similarity()

        for seen_movie_id in seen_movie_ids:
            seen_movie_ratings = ratings[ratings["movieId"] == seen_movie_id]
            try:
                similarity = movie_similarity_matrix[movie_list[seen_movie_id]][
                    movie_list[movie_id]]
            except Exception as e:
                similarity = 0.0

            similarity_movie = [seen_movie_id, similarity]
            similarity_movies.append(similarity_movie)  # 格式：[(10,0.91),(16,0.89),,,(231,0.23)]

            weight_similarity += similarity * \
                                 seen_movie_ratings[seen_movie_ratings["userId"] == user_index]["rating"].values[0]
            total_similarity += similarity
        result = weight_similarity / total_similarity
        results[movie_id] = result  # 电影id到评分的映射
    return results


def evaluate(user_sim_matrix, threshold):
    count = 0
    whole = 0
    for user_index in range(0, len(user_list)):
        for movie_index in range(0, len(movie_list)):
            predict = get_predict(user_sim_matrix, user_index, movie_index, 10)
            if ratings_matrix[user_index][movie_index] != 0:  # 原来有评分的话
                whole += 1
                if abs(predict - ratings_matrix[user_index][movie_index]) < threshold:
                    count += 1
    print("模型准确率为：" + count / whole)


if __name__ == '__main__':
    output = [['userId', 'movieId']]
    for user in user_list:
        res1 = userCF_recommend(user_sim_matrix, user_map[user], 10, 1)[0]
        res2 = itemCF_recommend(user_map[user], user_favor_matrix, type_rank_map)
        if res1 in content_based_list(user):
            output.append([user, movie_map_reverse[res1[0]]])
        if res2 in content_based_list(user):
            if res1 != res2:
                output.append([user, movie_map_reverse[res2[0]]])

    with open("movie_recommend.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["userId", "movieId"])
        for user_id in output.keys():
            for recommend_id in output[user_id]:
                writer.writerow([user_id, recommend_id])

    for threshold in [0.5, 0.75, 1]:
        evaluate(user_sim_matrix, threshold)
        # 评估结果
