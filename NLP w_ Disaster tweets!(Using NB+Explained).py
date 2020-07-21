# Importing all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import re
import string
import nltk
from nltk.corpus import stopwords
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from wordcloud import WordCloud,STOPWORDS
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
from nlppreprocess import NLP
from sklearn.feature_extraction.text import CountVectorizer


## Loading all the data
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
dataset=pd.concat([train,test])

## For exploring of data set , refer this kernel https://www.kaggle.com/friskycodeur/nlp-w-disaster-tweets-explained

# Text Pre-processing

def lowercase_text(text):
    return text.lower()

train.text=train.text.apply(lambda x: lowercase_text(x))
test.text=test.text.apply(lambda x: lowercase_text(x))

def remove_noise(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

train.text=train.text.apply(lambda x: remove_noise(x))
test.text=test.text.apply(lambda x: remove_noise(x))


## Using NLP Pre-processing 


nlp = NLP()

train['text'] = train['text'].apply(nlp.process)
test['text'] = test['text'].apply(nlp.process)  

## Stemming

stemmer = SnowballStemmer("english")

def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return ' '.join(text)

train['text'] = train['text'].apply(stemming)
test['text'] = test['text'].apply(stemming)


## Transforming words into vectors using Bag of Words

count_vectorizer=CountVectorizer(analyzer='word',binary=True)
count_vectorizer.fit(train.text)

train_vec = count_vectorizer.fit_transform(train.text)
test_vec = count_vectorizer.transform(test.text)

## Modelling 

y=train.target
model =MultinomialNB(alpha=1)
scores= model_selection.cross_val_score(model,train_vec,y,cv=6,scoring='f1')
model.fit(train_vec,y)

## Creating the submission file

sample_submission=pd.read_csv('../input/sample_submission.csv')
sample_submission.target= model.predict(test_vec)
sample_submission.to_csv('submission.csv',index=False)