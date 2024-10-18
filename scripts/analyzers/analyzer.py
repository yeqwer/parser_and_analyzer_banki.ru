import matplotlib.pyplot as plt
import pandas as pd
import json

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

#cleaning text
cl = Cleaner()
cleaned_train_texts = cl.clean_text(train_texts)
cleaned_test_texts = cl.clean_text(test_texts)

#set pipelines
pipelines = Pipelines.Linear_pipelines()
pipelines.pass_agress_classifier(cleaned_train_texts, train_ratings, cleaned_test_texts, test_ratings)

# words = cleaned_test_texts.to_string().split()
# df = pd.DataFrame(words, columns=['word'])
# count = df["word"].value_counts().reset_index()
# count.columns = ['word', 'frequency']
# top_10_words = count.head(10)

# plt.xticks(rotation=60)
# plt.plot(top_10_words["word"], top_10_words["frequency"])
# plt.show()

tfidf_matrix, feature_names = pipelines.tfidf_status(cleaned_train_texts)
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_sum = df_tfidf.sum(axis=0).reset_index()
tfidf_sum.columns = ['word', 'tfidf']
top_words = tfidf_sum.sort_values(by='tfidf', ascending=False).head(50)

plt.title("Наиболее важные слова в текстах")
plt.subplots_adjust(bottom=0.3)
plt.xticks(rotation=90)
plt.plot(top_words['word'], top_words["tfidf"])
plt.show()