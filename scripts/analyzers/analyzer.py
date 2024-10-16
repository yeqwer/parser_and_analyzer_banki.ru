import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy

#local import
from modules.text_cleaner import Cleaner 
from modules.analyze_pipelines import Pipelines

#set data for training
with open("./data/train_data.json", "r") as file:
  data = json.load(file)
  df = pd.DataFrame(data["responses"])
train_texts = df["TEXT"] #x_train
train_ratings = df["RATING"] #y_train
print(f"Total training data ={train_texts.shape}")

#set data for testing
with open("./data/test_data.json", "r") as file:
  data = json.load(file)
  df2 = pd.DataFrame(data["responses"])
test_texts = df2["TEXT"] #x_test
test_ratings = df2["RATING"] #y_test
print(f"Total testing data ={test_texts.shape}")
#total test data = 321 lines

#cleaning text
cl = Cleaner()
cleaned_train_texts = cl.clean_text(train_texts)
cleaned_test_texts = cl.clean_text(test_texts)

#set pipelines
pipelines = Pipelines()
pipelines.start_all_pipelines(cleaned_train_texts, train_ratings, cleaned_test_texts, test_ratings)

