import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
import shutil

class Plotter():
  def plot_most_important_words(self, pipelines, cleaned_train_texts):
    tfidf_matrix, feature_names = pipelines.tfidf_status(cleaned_train_texts)
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    tfidf_sum = df_tfidf.sum(axis=0).reset_index()
    tfidf_sum.columns = ['word', 'tfidf']
    top_words = tfidf_sum.sort_values(by='tfidf', ascending=False).head(40)

    fig, ax = plt.subplots(figsize=(12, 8))

    plt.title("Наиболее важные слова в текстах")
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=90)
    plt.plot(top_words['word'], top_words["tfidf"], color = "r")
    
    #save plot
    try: 
      shutil.rmtree(f'./tests/plots/most_important_words')
    except:
      pass
    
    if not (os.path.exists(f'./tests/plots/most_important_words')):
      os.makedirs(f'./tests/plots/most_important_words')

    plt.savefig('./tests/plots/most_important_words/most_imp_word_{}.png'.format(int(time.time())))

    #show plot
    # plt.show()

    plt.close()

  def plot_all_models_results(self, data):
    names = []
    trains = []
    tests = []

    for x in range(len(data)):
      for y in range(len(data[str(x)])):
        names.append(data[str(x)][str(y)]["model_name"])
        trains.append(round((data[str(x)][str(y)]["train_accuracy"]) * 100, 1))
        tests.append(round((data[str(x)][str(y)]["test_accuracy"]) * 100, 1))
    
    #plot specs
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(names))

    #plot axis
    b1 = ax.bar(x, trains, color="b", width=bar_width, label="Train data")
    b2 = ax.bar(x + bar_width, tests, color="r", width=bar_width, label="Test data")

    #plot style 
    ax.set_title('Точность моделей', pad=15, fontweight ='bold', fontsize = 15)
    ax.set_xlabel('Модели', fontweight ='bold', fontsize = 15)
    ax.set_ylabel('Процент точности (%)', fontweight ='bold', fontsize = 15)
    ax.set_ylim(60, 100)
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(names)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=30)
    plt.legend()

    #test over bars
    for bar in ax.patches:
      bar_value = bar.get_height()
      text = f'{bar_value:,}'
      text_x = bar.get_x() + bar.get_width() / 2
      text_y = bar.get_y() + bar_value
      bar_color = bar.get_facecolor()
      ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color,
              size=12)

    #save plot
    try: 
      shutil.rmtree(f'./tests/plots/models_results')
    except:
      pass

    if not (os.path.exists(f'./tests/plots/models_results')):
      os.makedirs(f'./tests/plots/models_results')

    plt.savefig('./tests/plots/models_results/model_results_{}.png'.format(int(time.time())))
    
    #show plot
    # plt.show() 

    plt.close()

  def plot_most_important_words_by_bank_names(self, pipelines, cleaned_train_texts, bank_name, review_count, positive):
    tfidf_matrix, feature_names = pipelines.tfidf_status(cleaned_train_texts)
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    tfidf_sum = df_tfidf.sum(axis=0).reset_index()
    tfidf_sum.columns = ['word', 'tfidf']
    top_words = tfidf_sum.sort_values(by='tfidf', ascending=False).head(40)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    if (positive):
      np = "положительное"
    else:
      np = "негативное"

    plt.title(f"Банк: {bank_name} | Количество отзывов: {review_count} | Настроение: {np}")
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=90)
    plt.plot(top_words['word'], top_words["tfidf"], color = "r")
    
    #save plot
    try: 
      if (positive):
        shutil.rmtree(f'./tests/plots/most_important_words_by_bank_names/{bank_name}/positive')
      else:
        shutil.rmtree(f'./tests/plots/most_important_words_by_bank_names/{bank_name}/negative')
    except:
      pass

    if not (os.path.exists(f'./tests/plots/most_important_words_by_bank_names/{bank_name}/positive')):
      os.makedirs(f'./tests/plots/most_important_words_by_bank_names/{bank_name}/positive')
    if not (os.path.exists(f'./tests/plots/most_important_words_by_bank_names/{bank_name}/negative')):
      os.makedirs(f'./tests/plots/most_important_words_by_bank_names/{bank_name}/negative')

    if(positive):
      plt.savefig(f'./tests/plots/most_important_words_by_bank_names/{bank_name}/positive/most_imp_words_{int(time.time())}.png')
    else:
      plt.savefig(f'./tests/plots/most_important_words_by_bank_names/{bank_name}/negative/most_imp_words_{int(time.time())}.png')

    #show plot
    # plt.show()

    plt.close()