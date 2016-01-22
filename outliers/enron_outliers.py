#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL")
#for point in data_dict:
   # if ((data_dict[point]['salary'] > 800000) or (data_dict[point]['bonus'] > 7000000)) and (data_dict[point]['salary'] != 'NaN' and data_dict[point]['bonus'] != 'NaN'):
        #print point
features = ["salary", "exercised_stock_options"]
data = featureFormat(data_dict, features)
scaler = MinMaxScaler()
rescaled_data = scaler.fit_transform(data)

scaler = preprocessing.MinMaxScaler().fit(data)
print scaler.transform([[200000., 1000000.]]) #rescales these two points based on the data. For Lesson 9 quiz
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
