# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 2021

@author: denise, nienke
"""

import numpy as np

import code_snippets
import parse_file
import preprocessing

from sklearn.neighbors import KNeighborsClassifier



def feature_extraction(X):
    features = []
    
    for x in X:
        features_for_x = []
        # Get maximum volume (amplitude) in the last ten percent of utterence
        ten_perc = int(0.1*len(x))
        features_for_x.append(np.max(x[:-ten_perc])) # max volume of last ten percent of the file
        features_for_x.append(np.max(x[:-ten_perc]) - np.max(x[ten_perc:])) # difference in max volume between last ten percent and first ten percent of sentence
    
        features.append(features_for_x)
    return np.matrix(features)

def run_knn(features, y):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(features, y)
    print(neigh.score(features,y))




X,y = preprocessing.split_soundfile()
features = feature_extraction(X)
run_knn(features, y)
