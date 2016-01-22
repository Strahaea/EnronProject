# -*- coding: utf-8 -*-
"""
Testing out minmax scaler
"""
from sklearn.preprocessing import MinMaxScaler
import numpy
weights = numpy.array([[115.],[140.],[175.]]) #need to be in this format
scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights) #applies formula to data
print rescaled_weight