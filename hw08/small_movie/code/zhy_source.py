# -*- coding: utf-8 -*-
# tags.csv:      userId，movieId，tags，timestamp
# ratings.csv:   userId，movieId，rating，timestamp
# Movies.csv:    movieId，title，genres

# -*- coding: utf-8 -*-
import csv
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity  # 计算余弦相似度


class MyRecommend:
    def __init__(self, movie_data_path, rating_data_path, tags_data_path):
        self.movie_data = pd.read_csv(movie_data_path)
        self.movie_data['genres'] = self.movie_data["genres"].fillna("")
        self.rating_data = pd.read_csv(rating_data_path)
        self.tags_data = pd.read_csv(tags_data_path)
        self.movie_origin_source = {}  # 建立movie Id-> ratings中的第i个出现顺序的映射
        self.user_average = {}  # 建立user_id ->  该用户打的平均分的映射（是除以电影总数）
        self.movie_average = {}  # 建立movie_id -> 该电影的平均分的映射

    def init_similarity(self):
        user_ids = self.rating_data["userId"].drop_duplicates().sort_values()
        movie_ids = self.rating_data["movieId"].drop_duplicates().sort_values()
        i = 0
        for movie_id in movie_ids:
            self.movie_origin_source[movie_id] = i
            i += 1

        # 以下是基于用户的协同过滤
        all_user_matrix = []  # 包含所有用户对所有电影的打分
        zeros = []
        for x in range(len(movie_ids)):
            zeros.append(0)
        for user_id in tqdm(user_ids):
            tmp = zeros.copy()
            seen_movies = self.rating_data[self.rating_data["userId"] == user_id]  # 该用户看过的电影的rating_data
            seen_movies_ids = seen_movies["movieId"]
            for seen_movies_id in seen_movies_ids:
                position = self.movie_origin_source[seen_movies_id]  # 某用户对某电影的rating，position存放该电影的位置
                tmp[position] = seen_movies[seen_movies["movieId"] == seen_movies_id]["rating"].values[0]
            average = np.mean(tmp)
            self.user_average[user_id] = average  # 记录该用户打分均值
            for i in range(len(tmp)):
                tmp[i] -= average  # tmp存放该用户打分之后减去用户对电影的均值评分
            # tmp里面包含该用户未看的电影
            all_user_matrix.append(tmp)

        all_user_df = pd.DataFrame(all_user_matrix)  # all_user_matrix
        all_user_matrix.clear()
        self.all_user_similarity = cosine_similarity(all_user_df.values)  # 求用户向量之间的相似度

        # 以下是基于物体的协同过滤
        all_movie_matrix = []
        zeros = []
        for x in range(len(user_ids)):
            zeros.append(0)
        for movie_id in tqdm(movie_ids):
            tmp = zeros.copy()
            seen_users = self.rating_data[self.rating_data["movieId"] == movie_id]
            seen_users_ids = seen_users["userId"]
            for seen_user_id in seen_users_ids:
                tmp[seen_user_id - 1] = seen_users[seen_users["userId"] == seen_user_id]["rating"].values[0]
            average = np.mean(tmp)  # 根据用户打分计算相似性
            self.movie_average[movie_id] = average  # 包含未打分用户，即记为0，平均分大概1.xx
            for i in range(len(tmp)):
                if tmp[i] != 0:  # 因为原来的处理将没有评分的默认为最差评我认为不妥
                    tmp[i] -= average  # 减去该用户对各电影评分的均值
            all_movie_matrix.append(tmp)
            ''' 相当于原来100名用户对其评分为(3,4,0,0,3,4,0...),平均分为1.5
                将其全部减去平均分，即(1.5,2.5,-1.5,-1.5,...),
                存在问题：如果未进行评价，则分数过低
            '''

        all_movie_df = pd.DataFrame(all_movie_matrix)  #
        all_movie_matrix.clear()
        self.all_movie_similarity = cosine_similarity(all_movie_df.values)

    def user_based_one(self, user_id, movie_id, is_seen=True):  # 1 1 false
        """
        用户-用户协同过滤
        预测 user_id 会给 movie_id 的评分
        :param is_seen:     是否看过该电影
        :param user_id:     某用户
        :param movie_id:    将要被该用户打分的电影
        :return:   预测的打分

        使用例子：user_based_one(user_id, recommend_id)          系统预测推荐电影的评分
                user_based_one(user_id, seen_movie_id, False)  系统预测已经看过电影的评分
        """
        # 用户 user_id 的全部评分信息
        user_ratings = self.rating_data[self.rating_data["userId"] == user_id]

        # 检查是不是没有评价过
        # 如果看过但是没有评价，则返回 0
        if len(user_ratings[user_ratings["movieId"] == movie_id].values) and is_seen:
            return 0.0

        # 看过影片 movie_id 的全部用户id
        seen_user_ids = self.rating_data[self.rating_data["movieId"] == movie_id]["userId"]
        # 用户相似度
        similarity_users = []
        weight_similarity = 0.0
        total_similarity = 0.0

        for seen_user_id in seen_user_ids:
            # 看过movie_id的电影的用户的全部评分情况
            seen_user_ratings = self.rating_data[self.rating_data["userId"] == seen_user_id]

            similarity = self.all_user_similarity[seen_user_id - 1][user_id - 1]  # 该用户与其他用户的相似度
            similarity_user = [seen_user_id, similarity]
            similarity_users.append(similarity_user)
            # 加权相似度的计算方式为用户之间的相似度乘以该用户（不是核心讨论用户）的评分
            weight_similarity += similarity * \
                                 seen_user_ratings[seen_user_ratings["movieId"] == movie_id]["rating"].values[0]
            total_similarity += similarity
        if total_similarity == 0.0 or total_similarity == 0:
            return 0.0
        # return weight_similarity / total_similarity + self.user_average[user_id]
        return weight_similarity / total_similarity

    '''
    该返回值存在问题，weight_similarity / total_similarity 已经是根据其相似的用户生成的预测评分，若 +self.user_average[user_id] 可能会超过5分
    目前尚未超过5分的原因是因为user_average[xxx]过小(因为用户的电影评分数量较少)
    '''

    def movie_based_one(self, user_id, movie_id, is_seen=True):
        """
        物品-物品协同过滤
        预测 user_id 会给 movie_id 的评分
        评分的依据是根据用户以往看过的电影，对每一部与其进行 similarity 的分析等
        :param is_seen:     是否看过该电影
        :param user_id:     某用户
        :param movie_id:    将要被该用户打分的电影
        :return:  预测的评分
        """
        # movie_id的全部评分情况
        movie_ratings = self.rating_data[self.rating_data["movieId"] == movie_id]

        # 检查是不是没有评价过
        if len(movie_ratings[movie_ratings["userId"] == user_id].values) and is_seen:
            return 0.0

        # user_id 看过的所有影片 movie_id
        seen_movie_ids = self.rating_data[self.rating_data["userId"] == user_id]["movieId"]

        # 影片相似度
        similarity_movies = []
        weight_similarity = 0.0
        total_similarity = 0.0

        for seen_movie_id in seen_movie_ids:
            seen_movie_ratings = self.rating_data[self.rating_data["movieId"] == seen_movie_id]
            try:
                similarity = self.all_movie_similarity[self.movie_origin_source[seen_movie_id]][
                    self.movie_origin_source[movie_id]]
            except Exception as e:
                similarity = 0.0

            similarity_movie = [seen_movie_id, similarity]
            similarity_movies.append(similarity_movie)  # 格式：[(10,0.91),(16,0.89),,,(231,0.23)]

            weight_similarity += similarity * \
                                 seen_movie_ratings[seen_movie_ratings["userId"] == user_id]["rating"].values[0]
            total_similarity += similarity
        result = weight_similarity / total_similarity
        # if movie_id not in self.movie_average:        # 如果电影列表中没有收集该电影，则返回 result？ 不能理解
        #     return result
        # return result + self.movie_average[movie_id]
        return result

    def content_based_predict(self, movie_ids, seen_movie_ids, n=10):
        """
        找到和 movie_id 相似的电影
        :param movie_ids:      # 用户最喜欢的前 n（10）部电影的id
        :param seen_movie_ids: 用户已经看过的电影
        :param n: 推荐个数
        :return:
        """
        tfidf = TfidfVectorizer(stop_words="english")  # 英语内建的停用词表
        tfidf_matrix = tfidf.fit_transform(self.movie_data["genres"])
        # 传入句，例如corpus =  ["I have a pen.","I have an apple."]
        # 一部电影可能有不同的分类，因此下面只能找到类型相似的电影（输入只有类型）

        # 计算余弦相似度
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # 建立电影列表索引，方便进行相互索引
        indices = pd.Series(self.movie_data.index, index=self.movie_data["movieId"]).drop_duplicates()

        similarities = {}                                           # 记录格式为：{740:0.999,1224:0.999}之类的

        for movie_id in movie_ids:
            idx = indices[movie_id]
            similarity = list(enumerate(cosine_sim[idx]))           # 存放的是[(0,0),(1,0)...(3,0.168)]数据，对于每一元组两个数，第一个表示movieId，第二个表示相似度
            similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
            similarity = similarity[1: 1 + int(n / 2)]              # 因为第0个similarity一定是自己
            for s in similarity:                                    # s形式为(740,0.998)，元组第一个数表示其他电影的idx，后一个数表示相似度
                if s[0] in similarities or s[0] in seen_movie_ids:  # 表示该电影已经和之前的电影有过相似的记录了，就直接跳过
                    continue
                else:
                    similarities[s[0]] = s[1]

        recommend_ids = list(similarities.keys())
        recommend_ids.sort(key=lambda x: similarities[x], reverse=True)  # 再总体按照相似度进行排序
        movie_indices = recommend_ids[:n]
        recommends = self.movie_data["movieId"].iloc[movie_indices]
        return recommends

    def predict(self, user_ids, n=10, least_rating=3.0):
        """
        联合推荐，先通过内容推荐来缩小范围，然后使用协同过滤
        :param user_ids: 用户范围
        :param n:内容过滤输出为前10
        :param least_rating:最低可以接受的评分
        :return:
        """
        users_result = {}
        for user_id in tqdm(user_ids):
            # 推荐的电影
            result = []
            seen_movie_ids = self.rating_data[self.rating_data["userId"] == user_id]["movieId"].drop_duplicates()
            # 用户最喜欢的前n（10）部电影的id
            love_movie_ids = self.rating_data[self.rating_data["userId"] == user_id].sort_values(by=["rating"]).head(n)[
                "movieId"].drop_duplicates()

            recommend_ids = self.content_based_predict(love_movie_ids, seen_movie_ids, n)
            for recommend_id in recommend_ids:
                result.append(recommend_id)

            # result是暂时的推荐 result
            # 进行协同过滤(物品和用户两种)，低于预测评分不进行推荐
            recommend_scores = []
            for recommend_id in recommend_ids:
                user_based_score = self.user_based_one(user_id, recommend_id)
                movie_based_score = self.movie_based_one(user_id, recommend_id)
                if user_based_score < least_rating and movie_based_score < least_rating:
                    continue
                recommend_scores.append([recommend_id, user_based_score, movie_based_score])
            recommend_scores.sort(key=lambda x: x[1] + x[2])

            # 推荐最多前2名
            if (len(recommend_scores) < 2):
                users_result[user_id] = [recommend_id[0] for recommend_id in recommend_scores]
            else:
                users_result[user_id] = [recommend_id[0] for recommend_id in recommend_scores[:2]]

        with open("../result/movie" + str(time.time()) + ".csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["userId", "movieId"])
            for user_id in users_result.keys():
                for recommend_id in users_result[user_id]:
                    writer.writerow([user_id, recommend_id])

    def evaluate(self, user_ids, edges):
        """
        评估效果，以一个用户为例，检验协同过滤算法
        假设用户对已看电影评分为 i，则判断 predict(i) 是否在 (i-edge,i+edge) 范围之内，以此确定评估效果。
        :param user_ids:
        :param edges: 边界
        :return:
        """
        valid = {}
        for edge in edges:
            valid[edge] = 0
        total = 0
        for user_id in user_ids:
            seen_movies = self.rating_data[self.rating_data["userId"] == user_id]
            seen_movie_ids = seen_movies["movieId"]
            total += len(seen_movie_ids)
            for seen_movie_id in tqdm(seen_movie_ids):
                real_score = seen_movies[seen_movies["movieId"] == seen_movie_id]["rating"].values[0]
                user_based_score = self.user_based_one(user_id, seen_movie_id, False)
                movie_based_score = self.movie_based_one(user_id, seen_movie_id, False)
                for edge in edges:
                    if (abs(real_score - user_based_score) <= edge or
                            abs(real_score - movie_based_score) <= edge):
                        valid[edge] += 1
        for edge in edges:
            print("上下浮动范围为" + str(edge) + "时的准确率为" + str(valid[edge] / total * 100) + "%")


if __name__ == '__main__':
    myRecommend = MyRecommend("../ml-latest-small/movies.csv",
                              "../ml-latest-small/ratings.csv",
                              "../ml-latest-small/tags.csv")
    myRecommend.init_similarity()
    print(myRecommend.user_based_one(1, 1, False))  # 测试用
    myRecommend.predict(range(611))
    myRecommend.evaluate(range(5), [0.25, 0.5, 0.75, 1])

# 预测流程：先根据 content_based 进行第一轮推荐，对这些推荐的电影进行 user_based_one 和 movie_based_one 获得评分，若评分小于3.0则绝对不予推荐，否则两个分数相加排名。
# 评估流程：以用户以往评分作为基准，根据 user_based_one 和 movie_based_one 获得评分。若评分于真实评分的上下浮动范围之内则可计数。
# content_based 是根据标签（tags）进行相似度推荐的， similarity 却是根据用户打分进行衡量的。

# 上下浮动范围指的是用户实际评分和预测评分之差
# 上下浮动范围为 0.25 时的准确率为33.72093023255814%
# 上下浮动范围为 0.5  时的准确率为62.4031007751938%
# 上下浮动范围为 0.75 时的准确率为80.62015503875969%
# 上下浮动范围为 1    时的准确率为88.17829457364341%

# 去掉 user_based_one 那个加上均分的内容
# 上下浮动范围为0.25时的准确率为34.49612403100775%
# 上下浮动范围为0.5时的准确率为61.04651162790697%
# 上下浮动范围为0.75时的准确率为80.42635658914729%
# 上下浮动范围为1时的准确率为87.59689922480621%

# 将未评分的电影设置为均分的结果（原代码将其设置为最差评价）
# 上下浮动范围为0.25时的准确率为22.674418604651162%
# 上下浮动范围为0.5时的准确率为46.89922480620155%
# 上下浮动范围为0.75时的准确率为80.81395348837209%
# 上下浮动范围为1时的准确率为87.20930232558139%
