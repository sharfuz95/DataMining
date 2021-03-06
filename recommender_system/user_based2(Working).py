#Helpful Resource
#https://www.dataquest.io/blog/k-nearest-neighbors-in-python/

import pandas as pd # pandas is a data manipulation library
import numpy as np #provides numerical arrays and functions to manipulate the arrays efficiently
from scipy.spatial import distance
import random
import math
from numpy.random import permutation
from sklearn.neighbors import KNeighborsRegressor


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
ratings_data = pd.read_csv('ml-latest-small/ratings.csv', sep=',')

print(ratings_data.shape)

#Ratings Columns:
#['userId' 'movieId' 'rating' 'timestamp']

#Remove timestamp column
ratings_data = ratings_data.drop(["timestamp"], axis = 1)

#Merge with Movie Data
ratings_data = ratings_data.merge(movies_data, left_on = "movieId", right_on = "movieId", how = "left")
#print(ratings_data.head(20))

#Movies rated by userId
selected_user = ratings_data[ratings_data["userId"] == 1]
print(selected_user.head(10))

#Check if movie has been rated by selected user
"""
for movie in selected_user["title"]:
    if movie == "Braveheart (1995)":
        print("---------------------------------- Watched")
    else:
        print("---------------------------------- Not Watched")
"""

distance_columns = ["rating"]

#Normalizing Ratings
ratings = ratings_data[distance_columns]
#print(ratings)
normalized_ratings = (ratings - ratings.mean()) / ratings.std()
#print(normalized_ratings)

user_normalized = normalized_ratings[ratings_data["userId"] == 1]

#Find the distance bewteen the selected user and everyone else
euclidean_distances = normalized_ratings.apply(lambda row: distance.euclidean(row, user_normalized), axis = 1)

#Create new Data Frame with distances
distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx":euclidean_distances.index })
distance_frame.sort_values("dist", inplace=True)

#Find the most similar user to the selected one
second_smallest = distance_frame.iloc[1]["idx"]
most_similar_to_selected = ratings_data.loc[int(second_smallest)]["userId"]


# Randomly shuffle the index of the ratings_data.
###random_indices = permutation(ratings_data.index)
# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
test_cutoff = math.floor(len(ratings_data)/3)
# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
test = ratings_data.loc[ratings_data.index[1:test_cutoff]]
# Generate the train set with the rest of the data.
train = ratings_data.loc[ratings_data.index[test_cutoff:]]

#print(train.head(20))
#print(test.head(20))

x_columns = ["userId", "movieId"]
y_column = ["rating"]

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(train[x_columns], train[y_column])
predictions = knn.predict(test[x_columns])

print(predictions)
# print("There are %i predictions" %len(predictions))
# print("There are %i tuples in the test set" %len(test))



comparison = test
comparison["predictions"] = predictions
print(comparison.head(20))

# Get the actual values for the test set.
actual = test[y_column]
# Compute the mean squared error of our predictions.
mse = (((predictions - actual) ** 2).sum()) / len(predictions)

print(mse)