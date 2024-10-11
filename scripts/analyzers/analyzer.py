import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with open("./data/learn_data.json", "r") as file:
  data = json.load(file)
  df = pd.DataFrame(data["responses"])

texts = df["TEXT"] #x
ratings = df["RATING"] #y

with open("./data/last_data.json", "r") as file:
  data = json.load(file)
  df2 = pd.DataFrame(data["responses"])

new_review = df2["TEXT"] #for test

cv = CountVectorizer()
tf = TfidfTransformer()
x_texts = cv.fit_transform(texts.values.astype('U'))
x_texts_tf = tf.fit_transform(x_texts)
print(x_texts.shape)
print(cv.vocabulary_.get(u'топ'))

# mnb = MultinomialNB().fit(x_texts_tf, ratings)
sgdc = SGDClassifier().fit(x_texts_tf, ratings)

x_new_review = cv.transform(new_review)
x_new_review_tf = tf.transform(x_new_review)

predicted = sgdc.predict(x_new_review_tf)

for doc, category in zip(new_review, predicted):
  print('%r => %s \n' % (doc, category))
