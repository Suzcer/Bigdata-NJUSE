import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import *
from tqdm import tqdm


class MovieLensSys:
    def __init__(self):
        self.movieSource = pd.read_csv("../ml-latest-small/movies.csv", encoding='gbk')
        self.movieSource['genres'] = self.movieSource["genres"].fillna("")
        self.ratingSource = pd.read_csv("../ml-latest-small/ratings.csv")
        self.tagsSource = pd.read_csv("../ml-latest-small/tags.csv")
        self.movieIndex = {}  # 建立movie Id-> ratings中的第i个出现顺序的映射
        self.allUserRatings = []
        self.ratingMovies = {}
        self.userPreferMovies = {}
        self.userIds = self.ratingSource["userId"].drop_duplicates().sort_values()
        self.movieIds = self.ratingSource["movieId"].drop_duplicates().sort_values()

        i = -1
        for movieId in self.movieIds:
            self.movieIndex[movieId] = ++i

        for userId in self.userIds:
            self.ratingMovies[userId] = self.ratingSource[self.ratingSource["userId"] == userId]
            self.userPreferMovies[userId] = \
                self.ratingSource[self.ratingSource["userId"] == userId].sort_values(by=["rating"]).head(8)[
                    "movieId"].drop_duplicates()

    def predict(self):
        usersResult = {}
        for userId in tqdm(self.userIds):
        # for userId in tqdm(range(1,6)):
            ratingMovieIds = self.ratingMovies[userId]['movieId']
            userPreferMovies = self.userPreferMovies[userId]

            possible_ids = self.contentBased(userPreferMovies, ratingMovieIds)
            possibleLists = []
            for possibleId in possible_ids:
                userCF = self.userCF(userId, possibleId)
                possibleLists.append([possibleId, userCF])
            possibleLists.sort(key=lambda x: x[1], reverse=True)

            usersResult[userId] = [_[0] for _ in possibleLists[:1]]

        self.save(usersResult)

    def userSim(self):
        for userId in tqdm(self.userIds):
            userRatings = [0] * len(self.movieIds)
            ratingMovies = self.ratingMovies[userId]
            for ratingMovieId in ratingMovies["movieId"]:
                position = self.movieIndex[ratingMovieId]  # 某用户对某电影的rating，position存放该电影的位置
                userRatings[position] = ratingMovies[ratingMovies["movieId"] == ratingMovieId]["rating"].values[0]
            average = np.mean(userRatings)
            map(lambda x: x - average, userRatings)
            self.allUserRatings.append(userRatings)
        self.userSims = cosine_similarity(pd.DataFrame(self.allUserRatings).values)  # 求用户向量之间的相似度

    def save(self, Results):
        with open("../result/movie.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["userId", "movieId"])
            for userId in Results.keys():
                for movieId in Results[userId]:
                    writer.writerow([userId, movieId])

    def getUserRating(self, userId, movieId):
        rating_ = \
            self.ratingSource[
                (self.ratingSource["userId"] == userId) & (self.ratingSource["movieId"] == movieId)][
                "rating"].values[0]
        return rating_

    def userCF(self, userId, movieId):
        # 判断是否看过该电影
        user_ratings = self.ratingSource[self.ratingSource["userId"] == userId]
        if len(user_ratings[user_ratings["movieId"] == movieId].values) != 0:
            return 0.0

        weightSum = 0.0
        totalSum = 0.0
        ratingUserIds = self.ratingSource[self.ratingSource["movieId"] == movieId]["userId"]  # 看过电影 movie_id 的全部用户id
        for eachId in ratingUserIds:
            # 获取评分；获取相似度；分别计算
            seen_user_rating = self.getUserRating(eachId, movieId)
            similarity = self.userSims[eachId - 1][userId - 1]
            weightSum += similarity * seen_user_rating
            totalSum += similarity

        if totalSum == 0.0:
            return 0.0
        return weightSum / totalSum

    def contentBased(self, movieIds, ratingMovieIds):
        genresSim = self.tfidf()
        indices = pd.Series(self.movieSource.index,
                            index=self.movieSource["movieId"]).drop_duplicates()  # 建立电影列表索引，方便进行相互索引

        similarities = {}  # 记录格式为：{740:0.999,1224:0.999}之类的
        for movieId in movieIds:
            index = indices[movieId]
            similarity = list(
                enumerate(genresSim[index]))  # 存放的是[(0,0),(1,0)...(3,0.168)]数据，对于每一元组两个数，第一个表示movieId，第二个表示相似度
            for s in sorted(similarity, key=lambda x: x[1], reverse=True)[
                     1:10]:  # s形式为(740,0.998)，元组第一个数表示其他电影的idx，后一个数表示相似度
                if (s[0] not in similarities) and (s[0] not in ratingMovieIds):
                    similarities[s[0]] = s[1]
        ids = list(similarities.keys())

        ids.sort(key=lambda x: similarities[x], reverse=True)
        movieIndex = ids[:5]
        return self.movieSource["movieId"].iloc[movieIndex]

    def tfidf(self):
        tfidfMatrix = TfidfVectorizer(stop_words="english").fit_transform(self.movieSource["genres"])
        genresSim = linear_kernel(tfidfMatrix, tfidfMatrix)  # 计算余弦相似度
        return genresSim


if __name__ == '__main__':
    sys = MovieLensSys()
    sys.userSim()
    sys.predict()
