#Resource followed for the analysis
#https://www.kaggle.com/jneupane12/analysis-of-movielens-dataset-beginner-sanalysis

import pandas as pd # pandas is a data manipulation library
import numpy as np #provides numerical arrays and functions to manipulate the arrays efficiently
import random
import matplotlib.pyplot as plt # data visualization library
from wordcloud import WordCloud, STOPWORDS #used to generate world cloud


#####################
#Movie Data Analysis#
#####################

#Read file
data = pd.read_csv('ml-latest-small/movies.csv', sep=',')

print("The Movies Dataset has %i rows and %i columns" %(data.shape))
print("Here is a sample of the data:\n")
print(data.head())

#Check if any row is null
print("\nCheck if any row is null:")
print(data.isnull().any())

#Verify number of unique movies: 9742  
movies = data['movieId'].unique().tolist()
print("\nNumber of unique movies: %i" %len(movies))


"""
#Fltering movies per genre
action_movies = data['genres'].str.contains('Action')
adventure_movies = data['genres'].str.contains('Adventure')
animation_movies = data['genres'].str.contains('Animation')
children_movies = data['genres'].str.contains('Children')
comedy_movies = data['genres'].str.contains('Comedy')
crime_movies = data['genres'].str.contains('Crime')
documentary_movies = data['genres'].str.contains('Documentary')
drama_movies = data['genres'].str.contains('Drama')
fantasy_movies = data['genres'].str.contains('Fantasy')
noir_movies = data['genres'].str.contains('Film-Noir')
horror_movies = data['genres'].str.contains('Horror')
musical_movies = data['genres'].str.contains('Musical')
mystery_movies = data['genres'].str.contains('Mystery')
romance_movies = data['genres'].str.contains('Romance')
scifi_movies = data['genres'].str.contains('Sci-Fi')
thriller_movies = data['genres'].str.contains('Thriller')
war_movies = data['genres'].str.contains('War')
western_movies = data['genres'].str.contains('Western')
nogenre_movies = data['genres'].str.contains('no genres listed')

#Count number of movies per Genre
print("\n\nNumber of 'Action' movies: %i" %len(data[action_movies]))
#print(data[action_movies])

"""

#Define a function that counts the number of times each genre appear:
def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue
        for s in liste_keywords: 
            if pd.notnull(s): keyword_count[s] += 1
    #Convert the dictionary in a list to sort the keywords by frequency
    keyword_occurrences = []
    for k,v in keyword_count.items():
        keyword_occurrences.append([k,v])
    keyword_occurrences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurrences, keyword_count

#Make census of the genres:
genre_labels = set()
for s in data['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))


#Number of unique genres: 20
keyword_occurrences, dum = count_word(data, 'genres', genre_labels)
print("\nNumber of unique genres: %i" %len(keyword_occurrences))

#Display how many times each genre occurs:
print("\nGenre Frequency:")
for i in keyword_occurrences:
    print(i)


## DATA VISUALIZATION ##

#Function that controls the color of the words
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    #Red Hue
    h = int(0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)


#Display the result as a wordcloud
words = dict()
trunc_occurences = keyword_occurrences
for s in trunc_occurences:
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

#Display the same result as a histogram
fig = plt.figure(1, figsize=(36,26))
ax2 = fig.add_subplot(1,1,1)
y_axis = [i[1] for i in trunc_occurences]
x_axis = [k for k,i in enumerate(trunc_occurences)]
x_label = [i[0] for i in trunc_occurences]
plt.xticks(rotation=85, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(x_axis, x_label)
plt.ylabel("No. of occurences", fontsize = 16, labelpad = 0)
plt.xlabel("Genres", fontsize = 16, labelpad = 0)
ax2.bar(x_axis, y_axis, align = 'center', color='r')
plt.title("Distribution of Genres",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 30)
plt.show()


