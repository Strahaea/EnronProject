#!/usr/bin/python
"""
Analyzes the Enron dataset to find perons of interest
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
#
#incomplete feature list, oversized now til more selection is done

features_list = ["poi", "salary", "to_messages", "deferral_payments", "total_payments",
                 "exercised_stock_options", "bonus", "restricted_stock", 
                 "shared_receipt_with_poi", "restricted_stock_deferred", "total_stock_value",
                 "expenses", "loan_advances", "from_messages", "from_this_person_to_poi",
                 "director_fees", "deferred_income", "long_term_incentive", "from_poi_to_this_person"]



### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


### Outlier Removal
"""
Found an extreme outlier, the total salaries, in salary and removed it
as it was not represenative of the data, after removal all points appear valid
"""
del data_dict["TOTAL"]

"""
Graph to check for outliers

for point in data_dict:
    person_info = data_dict[point]
    salary = person_info["salary"]
    bonus = person_info["bonus"]
    plt.scatter( salary, bonus )
plt.xlabel("salary")
plt.pyplot.ylabel("bonus")
plt.pyplot.show()
"""

### New Features

my_dataset = data_dict #rename the data dictionary

### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)

#Feature scaling to improve accuracy
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
rescaled_features = scaler.fit_transform(np.array(features)) #rescales data


#break labels and features into training and testing sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


#Basic plan to find persons of interest:

#Select Features (outlier removal here?) - Lesson 11



#Run a PCA which will compress the features


#Remove outlier at some point (earlier?)

#Run Decision Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40) #make the classifier
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

#Playing with features to see which to use/feature selection
#print clf.feature_importances_
#print np.argmax(clf.feature_importances_)
#print features_list[11] #because shifted over 1 because no poi

"""
when run with this feature list
features_list = ["poi", "salary", "to_messages", "deferral_payments", "total_payments",
                 "exercised_stock_options", "bonus", "restricted_stock", 
                 "shared_receipt_with_poi", "restricted_stock_deferred", "total_stock_value",
                 "expenses", "loan_advances", "from_messages", "other", "from_this_person_to_poi",
                 "director_fees", "deferred_income", "long_term_incentive", "from_poi_to_this_person"]
had an 86% accuracy with other doing 90% and expenses doing 10% 
when these were removed and the feature importances were checked again
  
"""
#Find accuracy
from sklearn.metrics import accuracy_score #now that we have our prediction see how accurate it is
accuracy = accuracy_score(pred, labels_test)
print accuracy





### dump classifier, dataset and features_list so 
### anyone can run/check results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )


