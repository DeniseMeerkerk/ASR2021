import numpy as np

import code_snippets
import parse_file
import preprocessing


X,y = preprocessing.split_soundfile()

def feature_extraction(X):
    features = []
    
    for x in X:
        features_for_x = []
        # Get maximum volume (amplitude) in the last ten percent of utterence
        last_ten_perc = int(0.1*len(x))
        print(x)
        print(last_ten_perc)
        features_for_x.append(np.max(x[:-last_ten_perc]))
    
        features.append(features_for_x)
    return features


print(np.max(X[0]), np.max(X[1]), y)

print(len(X[0]), len(X[1]))

print(feature_extraction(X))
