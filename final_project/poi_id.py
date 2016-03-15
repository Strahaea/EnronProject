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

features_list = ["poi",  "from_messages", "restricted_stock_deferred", "director_fees" ]



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

### New Features to be added later

my_dataset = data_dict #rename the data dictionary for new features

### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)

#Feature scaling to improve accuracy, currently not implemented due to issue with predictor
"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
rescaled_features = scaler.fit_transform(np.array(features)) #rescales data
"""

#break labels and features into training and testing sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(rescaled_features, labels, test_size=0.3, random_state=42)


#Run a PCA which will compress the features


#Run Decision Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40) #make the classifier
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

#Playing with features to see which to use/feature selection
print clf.feature_importances_
best_feature_pos = np.argmax(clf.feature_importances_)
print features_list[best_feature_pos +1] #This will be the name of the most important feature, +1 cause no poi

"""
#Keeping track of best features, features importance in order is:
"expenses",  "salary","restricted_stock", "total_stock_value", "bonus", "exercised_stock_options", 
"shared_receipt_with_poi",  "long_term_incentive","total_payments", "from_this_person_to_poi",
"loan_advances","deferred_income", "deferral_payments", "to_messages",  "from_messages", 
"restricted_stock_deferred", "director_fees"

#Noticed getting a signicantly higher accuracy with from_messages, director_fees, and restriced_stock_deferred
#94% accuracy with those
"""
#Find accuracy
from sklearn.metrics import accuracy_score #now that we have our prediction see how accurate it is
accuracy = accuracy_score(pred, labels_test)
print accuracy #Note if some people who are persons of interest got away we shouldn't expect 100% accuracy

"""
Now Using our Classifier see if there are any persons of interest,
while not initially given as such, who are identified as such by
our classifier, print their names
"""

### dump classifier, dataset and features_list so 
### anyone can run/check results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )


