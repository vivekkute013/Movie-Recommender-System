# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:18:10 2022

@author: Vivek
"""

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

movies = pd.read_csv("D:\Machine Learning\dataset_files/tmdb_5000_credits.csv")
credit = pd.read_csv("D:\Machine Learning\dataset_files/tmdb_5000_movies.csv")

print(movies.head(1))
print(credit.head(1))

print(movies.shape)
print(credit.shape)

'''Merge both the datasets'''
movies = movies.merge(credit,on="title")
print(movies.shape)

call = movies.info()
'''
Important Column
Genres
ID
keywords
titles
overview
cast
crew
'''
movies = movies[['movie_id','title','overview','cast','crew','genres','keywords']]

'''We will make 3 colums of all, movie_id + title + tag
tag is combination of all other columns.
for that we have to preprocess that data.'''

print(movies.isnull().sum())

''' drop the null portions'''

movies.dropna(inplace=True)
print(movies.isnull().sum())

'''TO remove duplicated data'''
print(movies.duplicated().sum())

print(movies.iloc[0].genres)

'''
to convert this 
[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
                    to 
["Action","Adventure","Fantasy","Sci-Fi"]

This Is a Helper Function
But we have to convert string into List to run the for loop, so 
we use ast.literal_eval()'''
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
        
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

''' For Cast we want First 3 actors'''
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)

''' As we have to extract name from the dictionary who's job has Director as value.'''
def director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(director)

''' To convert overview into list'''
movies['overview'] = movies['overview'].apply(lambda x:x.split())

''' To remove the extra spaces as they can confuse the model.'''
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

'''Converting similar words into one word using NLP'''
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)

new_df['tags'].apply(stem)

'''Vectorising'''
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
''' Also convert it into a numpy array'''
vectors = cv.fit_transform(new_df['tags']).toarray()

'''words which occured most frequently are'''
freq = cv.get_feature_names()

'''To calculate distance between two vwctors there are two methods - Euclidian and cosine.
             We are going to use eeuclian'''

from sklearn.metrics.pairwise import cosine_similarity
Similarity = cosine_similarity(vectors)

sorted(list(enumerate(Similarity[0])), reverse= True, key =lambda x:x[1])

''' Function '''
def recommend(movies):
    movies_index =  new_df[new_df['title'] == movie].index[0]
    distance = Similarity[movies_index]
    movies_list = sorted(list(enumerate(Similarity[0])), reverse= True, key =lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(i[0])

print(recommend('Avatar'))












