from modules.text_cleaner import Cleaner #local import
import json
import pandas as pd
from pymorphy3 import MorphAnalyzer

with open("./data/clean_data.json", "r") as file:
  data = json.load(file)
  df = pd.DataFrame(data["responses"])
train_texts = df["TEXT"] #x_train
train_ratings = df["RATING"] #y_train

cl = Cleaner()
cleaned_train_texts = cl.clean_text(train_texts)

# morph = MorphAnalyzer()
# print(morph.normal_forms("Играют"))