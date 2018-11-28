#Resource followed for the analysis
#https://www.kaggle.com/jneupane12/analysis-of-movielens-dataset-beginner-sanalysis

import pandas as pd # pandas is a data manipulation library
import numpy as np #provides numerical arrays and functions to manipulate the arrays efficiently
import random
import matplotlib.pyplot as plt # data visualization library
from wordcloud import WordCloud, STOPWORDS #used to generate world cloud


#######################
#Ratings Data Analysis#
#######################

#Read files
ratings_data=pd.read_csv('ml-latest-small/ratings.csv',sep=',')

print("The Ratings Dataset has %i rows and %i columns" %(ratings_data.shape))
print("Here is a sample of the data:\n")
print(ratings_data.head())

#Check if any row is null
print("\nCheck if any row is null:")
print(ratings_data.isnull().any())

#Summary of ratings.csv
print("\nHere are the statistics of Rating:\n")
print(ratings_data['rating'].describe())
