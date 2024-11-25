import pandas as pd
import json

#local import
from modules.text_cleaner import Cleaner 
from modules.analyze_pipelines import Pipelines
from modules.plots_creator import Plotter

#set data for training
with open("./data/train_data.json", "r", encoding='utf-8') as file:
  data = json.load(file)
  df_train = pd.DataFrame(data["responses"])
df_train_bank_names = df_train["BANK NAME"]
train_texts = df_train["TEXT"] #x_train
train_ratings = df_train["RATING"] #y_train
print(f"Total training data ={train_texts.shape}")

#set data for testing
with open("./data/test_data.json", "r", encoding='utf-8') as file:
  data = json.load(file)
  df_test = pd.DataFrame(data["responses"])
df_test_bank_names = df_test["BANK NAME"]
test_texts = df_test["TEXT"] #x_test
test_ratings = df_test["RATING"] #y_test
print(f"Total testing data ={test_texts.shape}")

#find all bank names
df_bank_names = pd.concat([df_train, df_test], ignore_index=True)

#init libraries
plotter = Plotter()
cleaner= Cleaner()
pipelines = Pipelines()

#cleaning data
cleaned_train_texts = cleaner.clean_text(train_texts)
cleaned_test_texts = cleaner.clean_text(test_texts)

def start_analyze_data():
  result = pipelines.start_all_pipelines(cleaned_train_texts, train_ratings, cleaned_test_texts, test_ratings)

  #write to json
  with open("./data/models_results.json", "w", encoding='utf-8') as file:
    json.dump(result, file, indent=2)

def sort_to_bank_names():
  x = {"":[]}
  for index, row in df_bank_names.iterrows():
    current_bank = row["BANK NAME"]
    current_text = row["TEXT"]
    if(current_bank in x):
      x[current_bank].append(current_text)
    else:
      x[current_bank] = [current_text] 
  x.pop("")
  print(f"Total number of banks: {len(x)}")
  return x

def start_analyze_data_by_bank_names():
  x = sort_to_bank_names()
  for index, row in x.items():
    plotter.plot_most_important_words_by_bank_names(pipelines, cleaner.clean_text(pd.Series(row)), index, len(row))

def plot_models_results():
  #read models results in json
  with open("./data/models_results.json", "r") as file:
    data_models_results = dict(json.load(file))

  #create plot with models results
  plotter.plot_all_models_results(data_models_results)

def plot_most_imp_words(): 
  #create a plot with most important words before cleaning text
  plotter.plot_most_important_words(pipelines, train_texts)

  #create a plot with most important words after cleaning text
  plotter.plot_most_important_words(pipelines, cleaned_train_texts)

#start analyzers and plotter draws
# start_analyze_data()
# plot_models_results()
# plot_most_imp_words()
start_analyze_data_by_bank_names()