#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#print enron_data

#see how many people in data set
#print len(enron_data)

#see how many features in data set
#print len(enron_data['DERRICK JR. JAMES V'])

#find how many persons of interest in data set

count = 0
for person in enron_data:
    if enron_data[person]['salary'] != 'NaN':
        count = count + 1
print count

#see what a person of interest looks like
print enron_data['SKILLING JEFFREY K']

#Total Value of Stock belonging to James Prentice
print "James Prentice Total Stock Value is:"
print enron_data['PRENTICE JAMES']['total_stock_value']

#How many email messages do we have from Wesley Colwell to persons of interest?
print "Wesley Colwell sent this many messages to pois:"
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']

#What's the value of the stock options exercised by Jeffrey Skilling?
print "The value of Jeffrey Skilling's exercised stock options is:"
print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

#Who made the most money, Fastow, Lay, or Skilling?
print "Skilling: " + str(enron_data['SKILLING JEFFREY K']["total_payments"])
print "Lay: " + str(enron_data['LAY KENNETH L']["total_payments"])
print "Fastow: " + str(enron_data['FASTOW ANDREW S']['total_payments'])

