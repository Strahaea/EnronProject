#!/usr/bin/python


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
"""

import pickle
import sys

#import os
#os.getcwd()
#os.chdir('C:\Python27\Lib')
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split, KFold
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn import grid_search #grid search for testing different parameters to optimize the classifier
parameters = {'min_samples_split':[1, 400]}
svr = DecisionTreeClassifier() #make our tree classifier
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(features_train, labels_train) #fit it with the data
pred = clf.predict(features_test) #make predictions

#import numpy as np
from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred) #see the accuracy


### it's all yours from here forward!  


#Ideas: Feature Selection, Feture Scaling, PCA


#Basic plan to find persons of interest:

#Select Features (outlier removal here?) - Lesson 11
#Scale Features - Lesson 9
#cross-validation somewhere around here
#Run a PCA which will compress the features
#decision tree (why? justification?)
#Remove outlier at some point (earlier?)
#Run Decision Tree
#Do Validation bit
