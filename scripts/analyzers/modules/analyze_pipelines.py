from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

class Pipelines():
  
  def tfidf_status(self, data):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data)
    feature_names = tfidf.get_feature_names_out()
    return tfidf_matrix, feature_names
  
  #all pipelines starter
  def start_all_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
    naive_pipelines = self.Naive_pipelines()
    naive_pipelines.start_all_nb_pipelines(train_data_x, train_data_y, test_data_x, test_data_y)
    linear_pipelines = self.Linear_pipelines()
    linear_pipelines.start_all_linear_pipelines(train_data_x, train_data_y, test_data_x, test_data_y)
    vector_pipelines = self.Vector_pipelines()
    vector_pipelines.start_all_vector_pipelines(train_data_x, train_data_y, test_data_x, test_data_y)
    trees_pipelines = self.Trees_pipelines()
    trees_pipelines.start_all_trees_pipelines(train_data_x, train_data_y, test_data_x, test_data_y)
    neural_pipelines = self.Neural_pipelines()
    neural_pipelines.start_all_neural_pipelines(train_data_x, train_data_y, test_data_x, test_data_y)
#---------------------------------------------------------------------------------------------------
  class Naive_pipelines():
    
    #all starter
    def start_all_nb_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
      self.multinomial_nb(train_data_x, train_data_y, test_data_x, test_data_y)
      self.complement_nb(train_data_x, train_data_y, test_data_x, test_data_y)
      self.bernoulli_nb(train_data_x, train_data_y, test_data_x, test_data_y)

    #add pipeline for MultinomialNB model
    def multinomial_nb(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("mnb", MultinomialNB())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      print(
        f"\nMultinomialNB:"
        f"\n  train accuracy_score: {accuracy_score(train_data_y, predict_train)}"
        f"\n  test accuracy_score: {accuracy_score(test_data_y, predict_test)}"
        f"\n  test classification_report: {classification_report(test_data_y, predict_test, zero_division=0.0)}")
      
    #add pipeline for ComplementNB model
    def complement_nb(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("mnb", ComplementNB())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      print(
        f"\nComplementNB:"
        f"\n  train accuracy_score: {accuracy_score(train_data_y, predict_train)}"
        f"\n  test accuracy_score: {accuracy_score(test_data_y, predict_test)}"
        f"\n  test classification_report: {classification_report(test_data_y, predict_test, zero_division=0.0)}")
      
    #add pipeline for BernoulliNB model
    def bernoulli_nb(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("mnb", BernoulliNB())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      print(
        f"\nBernoulliNB:"
        f"\n  train accuracy_score: {accuracy_score(train_data_y, predict_train)}"
        f"\n  test accuracy_score: {accuracy_score(test_data_y, predict_test)}"
        f"\n  test classification_report: {classification_report(test_data_y, predict_test, zero_division=0.0)}")    
#---------------------------------------------------------------------------------------------------
  class Linear_pipelines():
    
    #all starter
    def start_all_linear_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
      self.sgd_classifier(train_data_x, train_data_y, test_data_x, test_data_y)
      self.logistic_regression(train_data_x, train_data_y, test_data_x, test_data_y)

    #add pipeline for SGDClassifier model
    def sgd_classifier(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("sgdc", SGDClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      print(
        f"\nSGDClassifier:"
        f"\n  train accuracy_score: {accuracy_score(train_data_y, predict_train)}"
        f"\n  test accuracy_score: {accuracy_score(test_data_y, predict_test)}"
        f"\n  test classification_report: {classification_report(test_data_y, predict_test, zero_division=0.0)}")
      
    #add pipeline for LogisticRegression model
    def logistic_regression(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("rfc", LogisticRegression())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      print(
        f"\nLogisticRegression:"
        f"\n  train accuracy_score: {accuracy_score(train_data_y, predict_train)}"
        f"\n  test accuracy_score: {accuracy_score(test_data_y, predict_test)}"
        f"\n  test classification_report: {classification_report(test_data_y, predict_test, zero_division=0.0)}")
      
    #add pipeline for PassiveAggressiveClassifier model
    def pass_agress_classifier(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipeline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("pac", PassiveAggressiveClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipeline.fit(train_data_x, train_data_y)
      predict_train = pipeline.predict(train_data_x)
      predict_test = pipeline.predict(test_data_x)
      print(
        f"\nPassiveAggressiveClassifier:"
        f"\n  train accuracy_score: {accuracy_score(train_data_y, predict_train)}"
        f"\n  test accuracy_score: {accuracy_score(test_data_y, predict_test)}"
        f"\n  test classification_report: {classification_report(test_data_y, predict_test, zero_division=0.0)}")
#---------------------------------------------------------------------------------------------------
  class Vector_pipelines():

    #all starter
    def start_all_vector_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
      self.linear_svc(train_data_x, train_data_y, test_data_x, test_data_y)

    #add pipeline for LinearSVC model
    def linear_svc(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("lsvc", LinearSVC())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipline.fit(train_data_x, train_data_y)
      predict_train = pipline.predict(train_data_x)
      predict_test = pipline.predict(test_data_x)
      print(
        f"\nLinearSVC:"
        f"\n  train accuracy_score: {accuracy_score(train_data_y, predict_train)}"
        f"\n  test accuracy_score: {accuracy_score(test_data_y, predict_test)}"
        f"\n  test classification_report: {classification_report(test_data_y, predict_test, zero_division=0.0)}")
#---------------------------------------------------------------------------------------------------   
  class Trees_pipelines():
    
    #all starter
    def start_all_trees_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
      self.random_forest(train_data_x, train_data_y, test_data_x, test_data_y)
      self.decision_tree(train_data_x, train_data_y, test_data_x, test_data_y)

    #add pipeline for RandomForestClassifier model
    def random_forest(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("rfc", RandomForestClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipline.fit(train_data_x, train_data_y)
      predict_train = pipline.predict(train_data_x)
      predict_test = pipline.predict(test_data_x)
      print(
        f"\nRandomForestClassifier:"
        f"\n  train accuracy_score: {accuracy_score(train_data_y, predict_train)}"
        f"\n  test accuracy_score: {accuracy_score(test_data_y, predict_test)}"
        f"\n  test classification_report: {classification_report(test_data_y, predict_test, zero_division=0.0)}")
      
    #add pipeline for DecisionTreeClassifier model
    def decision_tree(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("rtc", DecisionTreeClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipline.fit(train_data_x, train_data_y)
      predict_train = pipline.predict(train_data_x)
      predict_test = pipline.predict(test_data_x)
      print(
        f"\nDecisionTreeClassifier:"
        f"\n  train accuracy_score: {accuracy_score(train_data_y, predict_train)}"
        f"\n  test accuracy_score: {accuracy_score(test_data_y, predict_test)}"
        f"\n  test classification_report: {classification_report(test_data_y, predict_test, zero_division=0.0)}")
#---------------------------------------------------------------------------------------------------
  class Neural_pipelines():
    
    #all starter
    def start_all_neural_pipelines(self, train_data_x, train_data_y, test_data_x, test_data_y):
      self.mlp_classifier(train_data_x, train_data_y, test_data_x, test_data_y)

    #add pipeline for MLPClassifier model
    def mlp_classifier(self, train_data_x, train_data_y, test_data_x, test_data_y):
      pipline = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("mlpc", MLPClassifier())]) #("ss", StandardScaler(with_mean=False)), ("norm", Normalizer())
      pipline.fit(train_data_x, train_data_y)
      predict_train = pipline.predict(train_data_x)
      predict_test = pipline.predict(test_data_x)
      print(
        f"\nMLPClassifier:"
        f"\n  train accuracy_score: {accuracy_score(train_data_y, predict_train)}"
        f"\n  test accuracy_score: {accuracy_score(test_data_y, predict_test)}"
        f"\n  test classification_report: {classification_report(test_data_y, predict_test, zero_division=0.0)}")
#---------------------------------------------------------------------------------------------------