import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from modules.text_cleaner import Cleaner #local import

#set data for training
with open("./data/train_data.json", "r") as file:
  data = json.load(file)
  df = pd.DataFrame(data["responses"])
train_texts = df["TEXT"] #x_train
train_ratings = df["RATING"] #y_train
#total training data = 10501 lines

#set data for testing
with open("./data/test_data.json", "r") as file:
  data = json.load(file)
  df2 = pd.DataFrame(data["responses"])
test_texts = df2["TEXT"] #x_test
test_ratings = df2["RATING"] #y_test
#total test data = 321 lines

#clean the text
cl = Cleaner()
cleaned_train_texts = cl.clean_text(train_texts)
cleaned_test_texts = cl.clean_text(test_texts)

#add pipeline for SGDClassifier model
pipline_sgdc = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("sgdc", SGDClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
train_sgdc  = pipline_sgdc.fit(cleaned_train_texts, train_ratings)
predict_train_sgdc = pipline_sgdc.predict(cleaned_train_texts)
predict_test_sgdc = pipline_sgdc.predict(cleaned_test_texts)
print(f"\nSGDClassifier:"
      f"\n  train accuracy_score: {accuracy_score(train_ratings, predict_train_sgdc)}"
      f"\n  test accuracy_score: {accuracy_score(test_ratings, predict_test_sgdc)}")
      # f"\n  test precision_score: {f1_score(test_ratings, predict_test_sgdc)}")

#add pipeline for LogisticRegression model
pipline_lr = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("rfc", LogisticRegression())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
train_lr  = pipline_lr.fit(cleaned_train_texts, train_ratings)
predict_train_lr = pipline_lr.predict(cleaned_train_texts)
predict_test_lr = pipline_lr.predict(cleaned_test_texts)
print(f"\nLogisticRegression:"
      f"\n  train accuracy_score: {accuracy_score(train_ratings, predict_train_lr)}"
      f"\n  test accuracy_score: {accuracy_score(test_ratings, predict_test_lr)}")
      # f"\n  test precision_score: {f1_score(test_ratings, predict_test_lr)}")

#add pipeline for MultinomialNB model
pipline_mnb = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("mnb", MultinomialNB())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
train_mnb  = pipline_mnb.fit(cleaned_train_texts, train_ratings)
predict_train_mnb = pipline_mnb.predict(cleaned_train_texts)
predict_test_mnb = pipline_mnb.predict(cleaned_test_texts)
print(f"\nMultinomialNB:"
      f"\n  train accuracy_score: {accuracy_score(train_ratings, predict_train_mnb)}"
      f"\n  test accuracy_score: {accuracy_score(test_ratings, predict_test_mnb)}")
      # f"\n  test precision_score: {f1_score(test_ratings, predict_test_mnb)}")

#add pipeline for RandomForestClassifier model
pipline_rfc = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("rfc", RandomForestClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
train_rfc  = pipline_rfc.fit(cleaned_train_texts, train_ratings)
predict_train_rfc = pipline_rfc.predict(cleaned_train_texts)
predict_test_rfc = pipline_rfc.predict(cleaned_test_texts)
print(f"\nRandomForestClassifier:"
      f"\n  train accuracy_score: {accuracy_score(train_ratings, predict_train_rfc)}"
      f"\n  test accuracy_score: {accuracy_score(test_ratings, predict_test_rfc)}")
      # f"\n  test precision_score: {f1_score(test_ratings, predict_test_rfc)}")