from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
import pandas as pd
import nltk 
import re

nltk.download("stopwords")

class Cleaner():
  def __init__(self):
    self.stopwords = set(stopwords.words("russian"))
    self.morph = MorphAnalyzer()
    self.re = re

  def clean_text(self, text):
    text_am = text.dropna() #remove nulls objects
    text_am_lw = text_am.str.lower() #make all text to lower case
    text_am_lw_re = text_am_lw.apply(lambda x: self.re.sub(r'[^\w\s]', '', x)) #remove punctuation marks
    tokenized_text_am_lw_re = pd.Series(text_am_lw_re.str.split()) #tokenize text
    tokenized_text_am_lw_re_rsw = tokenized_text_am_lw_re.apply(lambda x: [word for word in x if word not in self.stopwords]) #remove stop words
    lemmatized_tokenized_text_am_lw_re_rsw = tokenized_text_am_lw_re_rsw.apply(lambda x: [self.morph.normal_forms(word)[0] for word in x]) #removing common morphological and inflectional endings
    text = lemmatized_tokenized_text_am_lw_re_rsw.astype('str')
    text = text.str.replace("'","")
    text = text.str.replace(",","")
    text = text.str.replace("[","")
    text = text.str.replace("]","")
    # print(f"\nAnti miss\n{text_am.to_list()}")
    # print(f"\nLower case\n{text_am_lw.to_list()}")
    # print(f"\nRemove marcs\n{text_am_lw_re.to_list()}")
    # print(f"\nTokenize\n{tokenized_text_am_lw_re.to_list()}")
    # print(f"\nRemove stopwords\n{tokenized_text_am_lw_re_rsw.to_list()}")
    # print(f"\nLemmatize\n{lemmatized_tokenized_text_am_lw_re_rsw.to_list()}")
    # print(f"\nTotal\n{text.to_list()}")
    return text 