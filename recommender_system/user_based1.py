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

def usersWhoWatchedMovie(movieId):
    return ratings_data.loc[ratings_data["movieId"] == movieId]

def set_pandas_options() -> None:
    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 1000
    pd.options.display.max_colwidth = 199
    pd.options.display.width = None
    # pd.options.display.precision = 2  # set as needed

set_pandas_options()


############
#User Based#
############

#Read file
movies_data = pd.read_csv('ml-latest-small/movies.csv', sep=',')
ratings_data=pd.read_csv('ml-latest-small/ratings.csv',sep=',')

#Remove timestamp column
ratings_data = ratings_data.drop(["timestamp"], axis = 1)

#print(movies_data.head())
#print(ratings_data.head())

ratings = ratings_data["rating"]
normalized_ratings = (ratings - ratings.mean()) / ratings.std()

print(normalized_ratings)

"""
print(moviesRatedByUser(1).head())
movieRating = moviesRatedByUser(1).iloc[0]
#print(movieRating)
print(movieRating[2])

movies_subset = moviesRatedByUser(1).head()["movieId"]



for movie in movies_subset:
    print(usersWhoWatchedMovie(movie))
    
"""
    
    