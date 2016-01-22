# -*- coding: utf-8 -*-
"""
Created on Fri May 08 21:11:28 2015

@author: Kenneth
"""
#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###



#next two lines dramatically reduce the amount of data to speed up the SVM
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100]
#store predictions

import numpy as np
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C = 10000)
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 
t0 = time()
print "C 10000"

clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)

print clf.predict(features_test[50])
"""
chris_emails = 0
for i in range(0, len(pred)):
    if pred[i] == 1:
        chris_emails = chris_emails + 1
print "We predict " + str(chris_emails) + " emails to have been written by Chris"
print chris_emails
"""

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print acc

#########################################################

