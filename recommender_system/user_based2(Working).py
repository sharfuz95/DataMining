#Helpful Resource
#https://www.dataquest.io/blog/k-nearest-neighbors-in-python/

import pandas as pd # pandas is a data manipulation library
import numpy as np #provides numerical arrays and functions to manipulate the arrays efficiently
from scipy.spatial import distance
import random
import math
from numpy.random import permutation
from sklearn.neighbors import KNeighborsRegressor



############
#User Based#
############

#Read file
movies_data = pd.read_csv('ml-latest-small/movies.csv', sep=',')
ratings_data=pd.read_csv('ml-latest-small/ratings.csv',sep=',')

#Ratings Columns:
#['userId' 'movieId' 'rating' 'timestamp']

#Remove timestamp column
ratings_data = ratings_data.drop(["timestamp"], axis = 1)

#Movies rated by userId
selected_user = ratings_data[ratings_data["userId"] == 1]
#print(selected_user)

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
random_indices = permutation(ratings_data.index)
# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
test_cutoff = math.floor(len(ratings_data)/3)
# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
test = ratings_data.loc[random_indices[1:test_cutoff]]
# Generate the train set with the rest of the data.
train = ratings_data.loc[random_indices[test_cutoff:]]

# print(train.head(20))
# print(test.head(20))

x_columns = ["userId", "movieId"]
y_column = ["rating"]

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(train[x_columns], train[y_column])
predictions = knn.predict(test[x_columns])

#print(predictions)

# Get the actual values for the test set.
actual = test[y_column]
# Compute the mean squared error of our predictions.
mse = (((predictions - actual) ** 2).sum()) / len(predictions)

print(mse)