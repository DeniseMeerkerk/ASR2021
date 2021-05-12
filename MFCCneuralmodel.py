#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:59:06 2021

@author: denise
"""
#%% packages
import librosa
import numpy as np
import preprocessing
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers

#%% audio input

def getMFCCData(part=1):
    # X,y = preprocessing.split_soundfile(-1) # one data file
    X,y = preprocessing.split_all_soundfiles(part) # 20% of data files
    
    sample_rate = 16000 # check this later
    feature_dim_1 = 20 # check this: n_mfcc=20
    
    mfcc_X=[]
    for x in X:
        mfcc_X.append(librosa.feature.mfcc(y=x, sr=sample_rate, S=None,
                                           n_mfcc=feature_dim_1, dct_type=2,
                                           norm='ortho', lifter=0))
    return mfcc_X, y

#2d padding
def MFCC_padding(mfcc_X):
    new_mfcc_X = mfcc_X.copy()
    new_mfcc_X.sort(key=np.shape, reverse=True)
    feature_dim_1 = mfcc_X[0].shape[0] # check this: n_mfcc=20
    feature_dim_2 = new_mfcc_X[0].shape[1] 
        
    xxx = np.zeros([len(mfcc_X),feature_dim_1,feature_dim_2])
    for n, dingen in enumerate(mfcc_X):
        l,k = dingen.shape
        xxx[n,:l,:k] = dingen
    return xxx, feature_dim_1, feature_dim_2

#%% model

def prepareMFCCdata(channel=1,part=1):
    mfcc_X, y = getMFCCData(part)
    
    xxx, feature_dim_1, feature_dim_2 = MFCC_padding(mfcc_X)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(xxx, y, test_size= 0.2,
                                                    random_state=True,
                                                    shuffle=True)
    # Reshaping to perform 2D convolution
    X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
    X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    class_weights = compute_class_weight('balanced',[0,1], y=y)
    
    return X_train, X_test, y_train, y_test, feature_dim_1, feature_dim_2, class_weights


def modelParameters():
    channel = 1
    epochs = 10
    batch_size = 32
    verbose = 1
    num_classes = 2
    return channel, epochs, batch_size, verbose, num_classes


#%% gestolen van: https://www.kaggle.com/ashirahama/simple-keras-cnn-with-mfcc


def get_model(channel,feature_dims,num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', 
                     input_shape=(feature_dims[0], feature_dims[1], channel)))
    #model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    #model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model

def evaluate(model, X_test, y_test, batch_size=32):
    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=32)
    y_test_pred = model.predict(X_test)
    cm=confusion_matrix(np.argmax(y_test,axis=1),np.argmax(y_test_pred,axis=1))
    print('confusion matrix:\n',cm)
    print("test loss, test acc:", results)
    return results, cm
#%% 

def main():
    part = 1
    channel, epochs, batch_size, verbose, num_classes = modelParameters()
    X_train, X_test, y_train, y_test, feature_dim_1, feature_dim_2, class_weights = prepareMFCCdata(channel,part=part)
    model = get_model(channel,[feature_dim_1,feature_dim_2],num_classes)
    
    optimizer = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.binary_crossentropy,optimizer=optimizer,
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
              verbose=verbose, validation_data=(X_test, y_test),
              class_weight={0:class_weights[0] , 1:class_weights[1]})
    
    results,cm = evaluate(model, X_test, y_test)
    return
#%%
if __name__ == '__main__':
    main()
#%% 