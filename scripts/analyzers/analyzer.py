import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

#add feature extraction for data
cl = Cleaner()
cv = CountVectorizer()
tf = TfidfTransformer()

#TODO here make a preparing data before vectorizing
cleaned_train_texts = cl.clean_text(train_texts)
cleaned_test_texts = cl.clean_text(test_texts)

#vectorizing train data
train_x_texts = cv.fit_transform(cleaned_train_texts.values.astype('U'))
train_x_texts_tf = tf.fit_transform(train_x_texts)

#vectorizing test data
test_x_texts = cv.transform(cleaned_test_texts)
test_x_texts_tf = tf.transform(test_x_texts)

#TODO here make a preparing data before training model

# ss = StandardScaler(with_mean=False).fit(train_x_texts_tf)
# nz = Normalizer().fit(train_x_texts_tf)
# print("BEFORE:")
# print(train_x_texts_tf)
# train_x_texts_tf_nz = nz.transform(train_x_texts_tf)
# train_x_texts_tf_nz_ss = ss.transform(train_x_texts_tf_nz)

# print("AFTER:")
# print(train_x_texts_tf_nz_ss)

#train SGDClassifier model
sgdc = SGDClassifier().fit(train_x_texts_tf, train_ratings)

#testing the SGDClassifie model 
predict_sgdc = sgdc.predict(test_x_texts_tf)

# for doc, category in zip(new_review, predicted):
#   print('%r => %s \n' % (doc, category))

print("no Pipeline: ")
print(accuracy_score(test_ratings, predict_sgdc))

#add pipeline for SGDClassifier model
pipline_sgdc = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("sgdc", SGDClassifier())])
train_sgdc  = pipline_sgdc.fit(cleaned_train_texts, train_ratings)
predict_sgdc = pipline_sgdc.predict(cleaned_test_texts)
print("Pipeline: ")
print(accuracy_score(test_ratings, predict_sgdc))