from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

import json

class Pipelines():
  
  def tfidf_status(self, data):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data)
    feature_names = tfidf.get_feature_names_out()
    return tfidf_matrix, feature_names
  
  #all pipelines starter
  def start_all_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
    naive_pipelines = self.Naive_pipelines()
    linear_pipelines = self.Linear_pipelines()
    vector_pipelines = self.Vector_pipelines()
    trees_pipelines = self.Trees_pipelines()
    neural_pipelines = self.Neural_pipelines()
    data = {0 :  naive_pipelines.start_all_nb_pipelines(train_data_x, train_data_y, test_data_x, test_data_y), 
            1 : linear_pipelines.start_all_linear_pipelines(train_data_x, train_data_y, test_data_x, test_data_y), 
            2 : vector_pipelines.start_all_vector_pipelines(train_data_x, train_data_y, test_data_x, test_data_y), 
            3 : trees_pipelines.start_all_trees_pipelines(train_data_x, train_data_y, test_data_x, test_data_y), 
            4 : neural_pipelines.start_all_neural_pipelines(train_data_x, train_data_y, test_data_x, test_data_y)}
    return data
#---------------------------------------------------------------------------------------------------
  class Naive_pipelines():
    
    #all starter
    def start_all_nb_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
      data = {0 : self.multinomial_nb(train_data_x, train_data_y, test_data_x, test_data_y), 
              1 : self.complement_nb(train_data_x, train_data_y, test_data_x, test_data_y),
              2 : self.bernoulli_nb(train_data_x, train_data_y, test_data_x, test_data_y)}
      return data

    #add pipeline for MultinomialNB model
    def multinomial_nb(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("mnb", MultinomialNB())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      data = {"model_name" : "MultinomialNB", "train_accuracy" : accuracy_score(train_data_y, predict_train), "test_accuracy" : accuracy_score(test_data_y, predict_test)}
      return data
      
    #add pipeline for ComplementNB model
    def complement_nb(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("mnb", ComplementNB())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      data = {"model_name" : "ComplementNB", "train_accuracy" : accuracy_score(train_data_y, predict_train), "test_accuracy" : accuracy_score(test_data_y, predict_test)}
      return data 
      
    #add pipeline for BernoulliNB model
    def bernoulli_nb(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("mnb", BernoulliNB())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      data = {"model_name" : "BernoulliNB", "train_accuracy" : accuracy_score(train_data_y, predict_train), "test_accuracy" : accuracy_score(test_data_y, predict_test)}
      return data
#---------------------------------------------------------------------------------------------------
  class Linear_pipelines():
    
    #all starter
    def start_all_linear_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
      data = {0 :  self.sgd_classifier(train_data_x, train_data_y, test_data_x, test_data_y), 
              1 : self.logistic_regression(train_data_x, train_data_y, test_data_x, test_data_y), 
              2 : self.pass_agress_classifier(train_data_x, train_data_y, test_data_x, test_data_y)}
      return data

    #add pipeline for SGDClassifier model
    def sgd_classifier(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("sgdc", SGDClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      data = {"model_name" : "SGD Classifier", "train_accuracy" : accuracy_score(train_data_y, predict_train), "test_accuracy" : accuracy_score(test_data_y, predict_test)}
      return data
    
    #add pipeline for LogisticRegression model
    def logistic_regression(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("rfc", LogisticRegression())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      data = {"model_name" : "Logistic Regression", "train_accuracy" : accuracy_score(train_data_y, predict_train), "test_accuracy" : accuracy_score(test_data_y, predict_test)}
      return data
    
    #add pipeline for PassiveAggressiveClassifier model
    def pass_agress_classifier(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("pac", PassiveAggressiveClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      data = {"model_name" : "Passive Aggressive Classifier", "train_accuracy" : accuracy_score(train_data_y, predict_train), "test_accuracy" : accuracy_score(test_data_y, predict_test)}
      return data
#---------------------------------------------------------------------------------------------------
  class Vector_pipelines():

    #all starter
    def start_all_vector_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
      data = {0 : self.linear_svc(train_data_x, train_data_y, test_data_x, test_data_y)}
      return data

    #add pipeline for LinearSVC model
    def linear_svc(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("lsvc", LinearSVC())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipline.fit(train_data_x, train_data_y)
      predict_train = pipline.predict(train_data_x)
      predict_test = pipline.predict(test_data_x)
      data = {"model_name" : "Linear SVC", "train_accuracy" : accuracy_score(train_data_y, predict_train), "test_accuracy" : accuracy_score(test_data_y, predict_test)}
      return data
#---------------------------------------------------------------------------------------------------   
  class Trees_pipelines():
    
    #all starter
    def start_all_trees_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
      data = {0 : self.random_forest(train_data_x, train_data_y, test_data_x, test_data_y), 
              1 : self.decision_tree(train_data_x, train_data_y, test_data_x, test_data_y)}
      return data

    #add pipeline for RandomForestClassifier model
    def random_forest(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("rfc", RandomForestClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipline.fit(train_data_x, train_data_y)
      predict_train = pipline.predict(train_data_x)
      predict_test = pipline.predict(test_data_x)
      data = {"model_name" : "Random Forest Classifier", "train_accuracy" : accuracy_score(train_data_y, predict_train), "test_accuracy" : accuracy_score(test_data_y, predict_test)}
      return data
      
    #add pipeline for DecisionTreeClassifier model
    def decision_tree(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("rtc", DecisionTreeClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipline.fit(train_data_x, train_data_y)
      predict_train = pipline.predict(train_data_x)
      predict_test = pipline.predict(test_data_x)
      data = {"model_name" : "Decision Tree Classifier", "train_accuracy" : accuracy_score(train_data_y, predict_train), "test_accuracy" : accuracy_score(test_data_y, predict_test)}
      return data 
#---------------------------------------------------------------------------------------------------
  class Neural_pipelines():
    
    #all starter
    def start_all_neural_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
      data = {0 : self.mlp_classifier(train_data_x, train_data_y, test_data_x, test_data_y)}
      return data

    #add pipeline for MLPClassifier model
    def mlp_classifier(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("mlpc", MLPClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipline.fit(train_data_x, train_data_y)
      predict_train = pipline.predict(train_data_x)
      predict_test = pipline.predict(test_data_x)
      data = {"model_name" : "MLP Classifier", "train_accuracy" : accuracy_score(train_data_y, predict_train), "test_accuracy" : accuracy_score(test_data_y, predict_test)}
      return data
#---------------------------------------------------------------------------------------------------