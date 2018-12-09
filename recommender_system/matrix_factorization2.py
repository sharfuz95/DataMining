import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SQLContext


sqlContext = SQLContext(sc)
# check if spark context is defined
print(sc.version)



ratings_data=pd.read_csv('ml-latest-small/ratings.csv',sep=',')

#Remove timestamp column
ratings_data = ratings_data.drop(["timestamp"], axis = 1)

X_train, X_test= ratings.randomSplit([0.8, 0.2])