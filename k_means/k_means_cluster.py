#!/usr/bin/python 

""" 
   k-means clustering mini-project

"""




import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than 4 clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )



for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
plt.show()

exstock = [] #making a list so can perform max and min functions on it
for person in data_dict:
    if data_dict[person][feature_1] != 'NaN': #appens all values not NaN
        exstock.append(data_dict[person][feature_1])


from sklearn.cluster import KMeans
features_list = ["poi", feature_1, feature_2]
data2 = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data2 )
clf = KMeans(n_clusters=2)
pred = clf.fit_predict( finance_features )
Draw(pred, finance_features, poi, name="clusters_before_scaling.pdf", f1_name=feature_1, f2_name=feature_2)


#rescaling stuff

def make_list(feature_name, data): #makes the features intoa workable list
    new_list = []
    for person in data:
        if data[person][feature_name] != 'NaN':
            new_list.append((data[person][feature_name])*1.0)
    return new_list
    
def resc(data): #rescales features from a list of them
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    rescaled = scaler.fit_transform(data) #applies formula to data
    return rescaled

def resc_simple(point, data):#for simple example/ dunno with sklearn
    return (point - min(data))/(max(data)-min(data)) 
    
salaries = make_list(feature_1, data_dict)
stocks = make_list(feature_2, data_dict)


print resc_simple(200000., salaries)
print resc_simple(1000000., stocks)

try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"





