import pandas as pd # pandas is a data manipulation library
import numpy as np #provides numerical arrays and functions to manipulate the arrays efficiently
from scipy.sparse.linalg import svds #package to do single value decomposition
import random
import matplotlib.pyplot as plt # data visualization library
from wordcloud import WordCloud, STOPWORDS #used to generate world cloud

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


#print(ratings_data.loc[ratings_data["userId"] == 1])
def moviesRatedByUser(userId):
    return ratings_data.loc[ratings_data["userId"] == userId]

def moviesNotRatedByUser(userId):
    return ratings_data.loc[ratings_data["userId"] != userId]

#print(moviesNotRatedByUser(1))



############
#User Based#
############

#Read file
movies_data = pd.read_csv('ml-latest-small/movies.csv', sep=',')
ratings_data=pd.read_csv('ml-latest-small/ratings.csv',sep=',')

# print(movies_data.head())
#print(ratings_data.head())

# remove timestamp column
ratings_data = ratings_data.drop(["timestamp"], axis = 1)

#pivot ratings around userId
# rating_pivot = ratings_data.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
rating_pivot = ratings_data.pivot(index = 'movieId', columns ='userId', values = 'rating').fillna(0)
#print(rating_pivot)

#transfrom into a sparse matrix for more efficient calculations
rating_matrix = csr_matrix(rating_pivot.values)
model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
model_knn.fit(rating_matrix)

#query_index = np.random.choice(rating_pivot.shape[0])
query_index = 1
print(rating_pivot.index[query_index])

distances, indices = model_knn.kneighbors(rating_pivot.iloc[query_index, :].reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print("Recommendations for {0}:\n".format(rating_pivot[query_index]))
    else:
        print("{0}: {1}, width distance of {2}:".format(i, rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))



