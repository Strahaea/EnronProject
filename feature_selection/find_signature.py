#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### the words (features) and authors (labels), already largely processed
words_file = "C:/Users/Kenneth/Desktop/Udacity/Intro_to_Machine_Learning/projects/ud120-projects/text_learning/your_word_data.pkl" ### you made this in previous mini-project
authors_file = "C:/Users/Kenneth/Desktop/Udacity/Intro_to_Machine_Learning/projects/ud421-projects/text_learning/your_email_authors.pkl"  ### this too
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (remainder go into training)
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train).toarray()
features_test  = vectorizer.transform(features_test).toarray()

print len(features_train)
### a classic way to overfit is to use a small number
### of data points and a large number of features
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]



### your code goes here
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score #to find the accuracy
print accuracy_score(labels_test, pred)

print numpy.argmax(clf.feature_importances_)
print max(clf.feature_importances_)
print vectorizer.get_feature_names()[21323]