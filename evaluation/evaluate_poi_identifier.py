#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
#this step splits the data into training and testing data


### it's all yours from here forward!  
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier() #make our tree classifier
clf.fit(features_train, labels_train) #fit it with the data
pred = clf.predict(features_test) #make predictions

#this step counts the number of true positives for precision and recall calculations
true_positives = 0 #counts number of times we guess poi and they were a poi
for i in range(0, len(pred)):
    if pred[i] == labels_test[i]:
        if pred[i] == 1. :
            true_positives = true_positives + 1
print true_positives

positives = 0 #counts number of times we predicted a poi
for i in range(0, len(pred)):
    if pred[i] == 1.:
        positives += 1
print positives

true_negatives = 0
for i in range(0, len(pred)):
    if pred[i] != labels_test[i]:
        if pred[i] == 1:
            true_negatives += 1
print true_negatives
#seeing the accuracy
from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred) #see the accuracy




