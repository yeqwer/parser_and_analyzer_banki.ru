from sklearn.model_selection import train_test_split
import pandas as pd
import json

#local import
from modules.text_cleaner import Cleaner 
from modules.analyze_pipelines import Pipelines
from modules.plots_creator import Plotter

#set data for training
with open("./data/data.json", "r", encoding='utf-8') as file:
  data = json.load(file)
  df = pd.DataFrame(data["responses"])
bank_names = df["BANK NAME"]
texts = df["TEXT"] #x_train
ratings = df["RATING"] #y_train

#data split
x_train, x_test, y_train, y_test = train_test_split(texts, ratings, test_size=0.30, random_state=42, shuffle=True)
print(f"\nTotal data to train: {len(x_train)}")
print(f"\nTotal data to test: {len(x_test)}\n")

#find all bank names
# df_bank_names = pd.concat([df_train, df_test], ignore_index=True)

#init libraries
plotter = Plotter()
cleaner= Cleaner()
pipelines = Pipelines()

#cleaning data
cleaned_x_train = cleaner.clean_text(x_train)
cleaned_x_test = cleaner.clean_text(x_test)

def start_analyze_data():
  result = pipelines.start_all_pipelines(cleaned_x_train, y_train, cleaned_x_test, y_test)

  #write to json
  with open("./data/models_results.json", "w", encoding='utf-8') as file:
    json.dump(result, file, indent=2)

def sort_to_bank_names():
  x = {"":[[]]}
  for index, row in df.iterrows():
    current_bank = row["BANK NAME"]
    current_text = row["TEXT"]
    current_rating = row["RATING"]
    if(current_bank in x):
      x[current_bank].append([current_text, current_rating])
    else:
      x[current_bank] = [[current_text, current_rating]] 

  x.pop("")

  bank_review = {"":0}
  for bank in x:
    bank_review[bank] = len(x[bank])
  bank_review.pop("")
  bank_review = sorted(bank_review.items(), key=lambda review: review[1], reverse=False)
  for i in bank_review:
    print(f"{i[0]} have {i[1]} reviews")
  print(f"\nTotal number of banks: {len(x)}\n")

  return x

def start_analyze_data_by_bank_names():
  x = sort_to_bank_names()

  for index, row in x.items():
    current_negative = []
    current_positive = []

    for r in row:
      if (int(r[1]) <= 3):
          current_negative.append(r[0])
      else:
          current_positive.append(r[0])

    if (current_positive):
      plotter.plot_most_important_words_by_bank_names(pipelines, cleaner.clean_text(pd.Series(current_positive)), index, len(current_positive), True)
    if (current_negative):
      plotter.plot_most_important_words_by_bank_names(pipelines, cleaner.clean_text(pd.Series(current_negative)), index, len(current_negative), False)  

    current_positive.clear()
    current_negative.clear()  

def plot_models_results():
  #read models results in json
  with open("./data/models_results.json", "r") as file:
    data_models_results = dict(json.load(file))

  #create plot with models results
  plotter.plot_all_models_results(data_models_results)

def plot_most_imp_words(): 
  #create a plot with most important words before cleaning text
  plotter.plot_most_important_words(pipelines, x_train)

  #create a plot with most important words after cleaning text
  plotter.plot_most_important_words(pipelines, cleaned_x_train)

#start analyzers and plotter draws
start_analyze_data()
plot_models_results()
plot_most_imp_words()
start_analyze_data_by_bank_names()