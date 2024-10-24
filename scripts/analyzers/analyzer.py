import matplotlib.pyplot as plt
import pandas as pd
import json

#local import
from modules.text_cleaner import Cleaner 
from modules.analyze_pipelines import Pipelines
from modules.plots_creator import Plotter

#set data for training
with open("./data/train_data.json", "r", encoding='utf-8') as file:
  data = json.load(file)
  df = pd.DataFrame(data["responses"])
train_texts = df["TEXT"] #x_train
train_ratings = df["RATING"] #y_train
print(f"Total training data ={train_texts.shape}")

#set data for testing
with open("./data/test_data.json", "r", encoding='utf-8') as file:
  data = json.load(file)
  df2 = pd.DataFrame(data["responses"])
test_texts = df2["TEXT"] #x_test
test_ratings = df2["RATING"] #y_test
print(f"Total testing data ={test_texts.shape}")

#set plotter
plotter = Plotter()

def start_analyze_data(): 
  #cleaning text
  cl = Cleaner()
  cleaned_train_texts = cl.clean_text(train_texts)
  cleaned_test_texts = cl.clean_text(test_texts)

  #set pipelines
  pipelines = Pipelines()
  result = pipelines.start_all_pipelines(cleaned_train_texts, train_ratings, cleaned_test_texts, test_ratings)
  print(result)

  #create plot with most important words
  plotter.plot_most_important_words(pipelines, cleaned_train_texts)

  #write to json
  with open("./data/models_results.json", "w", encoding='utf-8') as file:
    json.dump(result, file, indent=2)

#start analyzer
start_analyze_data()

#read models results in json
with open("./data/models_results.json", "r") as file:
  data_models_results = dict(json.load(file))

#create plot with models results
plotter.plot_all_models_results(data_models_results)