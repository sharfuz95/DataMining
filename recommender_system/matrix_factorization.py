import pandas as pd # pandas is a data manipulation library
import numpy as np #provides numerical arrays and functions to manipulate the arrays efficiently
from scipy.sparse.linalg import svds #package to do single value decomposition
import random
import matplotlib.pyplot as plt # data visualization library
from wordcloud import WordCloud, STOPWORDS #used to generate world cloud



######################
#Matrix Factorization#
######################

#Read file
movies_data = pd.read_csv('ml-latest-small/movies.csv', sep=',')
ratings_data=pd.read_csv('ml-latest-small/ratings.csv',sep=',')

pd.set_option('display.max_colwidth', -1)

# ratings_df = pd.DataFrame(ratings_data, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
# movies_df = pd.DataFrame(movies_data, columns = ['MovieID', 'Title', 'Genres'])
# movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)


#for title in movies_data["title"]:
#    print(title.encode("utf-8"))


#print(ratings_data.head())
#print(ratings_data.columns)

R_df = ratings_data.pivot(index = 'userId', columns ='movieId', values = 'rating')
R_df = R_df.fillna(R_df.mean())
#print(R_df.head())

R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

#print(R_demeaned)


U, sigma, Vt = svds(R_demeaned, k = 50)
#Make Sigma a diagonal matrix
sigma = np.diag(sigma)


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)

def recommend_movies(predictions_df, userID, movies_data, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    user_full = (user_data.merge(movies_data, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_data[~movies_data['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


already_rated, predictions = recommend_movies(preds_df, 1, movies_data, ratings_data, 10)

"""
Count number of movies rated by user
count = 0
for user in ratings_data["userId"]:
    if user == 1:
        count += 1

print ("User 1 rated %i movies" %count)
"""

print(already_rated.head(10))
print("\nPredictions")
print(predictions.head(10)[["title", "genres"]])