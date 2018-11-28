#Resource followed for the analysis
#https://www.kaggle.com/jneupane12/analysis-of-movielens-dataset-beginner-sanalysis

import pandas as pd # pandas is a data manipulation library
import numpy as np #provides numerical arrays and functions to manipulate the arrays efficiently
import random
import matplotlib.pyplot as plt # data visualization library
from wordcloud import WordCloud, STOPWORDS #used to generate world cloud


###################
#Tags Data Anlysis#
###################

#Read file
tags_data = pd.read_csv('ml-latest-small/tags.csv', sep=',')

print("The Tags Dataset has %i rows and %i columns" %(tags_data.shape))
print("Here is a sample of the data:\n")
print(tags_data.head())

#Check if any row is null
print("\nCheck if any row is null:")
print(tags_data.isnull().any())

#Get only "tag" column
tag_labels = tags_data['tag'] 

#Number of unique tags: 1589 
unique_tags = tags_data['tag'].unique().tolist()
print("\nNumber of unique tags: %i" %len(unique_tags))

#Count number of times a tag appears
tag_occurrences = []
for unique_tag in unique_tags:
    count = 0
    for tag in tag_labels:
        if unique_tag.lower() == tag.lower():
            count += 1
    tag_occurrences.append([unique_tag.lower(), count])

#Uncomment to Display how many times each tag occurs:
"""
print("Tag occurrences:")
for i in tag_occurrences:
    print(i)
"""

## DATA VISUALIZATION ##

#Function that controls the color of the words
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    #Blue Hue
    h = int(240)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)


#Show  the result as a wordcloud:
words = dict()
trunc_occurrences = tag_occurrences
for s in trunc_occurrences:
    words[s[0]] = s[1]
tone = 100 # define the color of the words
f, ax = plt.subplots(figsize=(28, 12))
wordcloud = WordCloud(width=2400,height=1600, background_color='white', 
                      max_words=1628,relative_scaling=0.7,
                      color_func = random_color_func,
                      normalize_plurals=False)
wordcloud.generate_from_frequencies(words)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Uncomment to Display the same result as a histogram
"""
fig = plt.figure(1, figsize=(26,26))
ax2 = fig.add_subplot(1,1,1)
y_axis = [i[1] for i in trunc_occurrences]
x_axis = [k for k,i in enumerate(trunc_occurrences)]
x_label = [i[0] for i in trunc_occurrences]
plt.xticks(rotation=85, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(x_axis, x_label)
plt.ylabel("No. of occurences", fontsize = 16, labelpad = 0)
plt.xlabel("Genres", fontsize = 16, labelpad = 0)
ax2.bar(x_axis, y_axis, align = 'center', color='r')
plt.title("Distribution of Tags",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 30)
plt.show()
"""