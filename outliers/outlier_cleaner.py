#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    cleaned_data = []
    for i in range(0, len(ages)):
        difference = (predictions[i][0] - net_worths[i][0])**2 #finds the difference
        cleaned_data.append([ages[i], net_worths[i], int(difference)])#appends random form of the difference
    # Sort the array by the 3rd tuple value
    cleaned_data.sort(key=lambda tup: tup[2])

    # Remove the 10% biggest values based on their residuals.
    cleaned_data = cleaned_data[:int(len(ages)*0.9)]
    return cleaned_data

"""
Plan:
Goal: Clean 10% of data
Find list of expected age, list of predicted ages
Make list of the max differences
Pop the items in the list until the length of the list is
90% of the length of the original list

Current problem solving how to find max in a tuple
"""
