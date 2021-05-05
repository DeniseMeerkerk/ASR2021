#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:21:18 2021

@author: denise
"""

#%% packages
import librosa
#import pandas as pd
#import os
#from PIL import Image
#import pathlib
#import csv 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import keras
from keras import layers
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')
#import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import code_snippets
import parse_file
import preprocessing


X,y = preprocessing.split_soundfile(-1)
padded_X = preprocessing.pad_data(X)

#%% waveplot of first (padded) segment
plt.figure(figsize=(14, 5))
librosa.display.waveplot(padded_X[0])

#%% ANN based on: https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
encoder = LabelEncoder()
y = encoder.fit_transform(y)
#Scaling the Feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(padded_X, dtype = float))#Dividing data into training and Testing set
X_train, X_test, y_train, y_test = train_test_split(padded_X, y, test_size=0.2)

#%% small dense model

model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%% fit the model
classifier = model.fit(X_train,
                    y_train,
                    epochs=10,
                    batch_size=32)

#%% evaluate the model
print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=32)
print("test loss, test acc:", results)

#%% evaluate on comp-a
"""WERKT ALLEEN NOG OP DENISE'S COMPUTER
MAAR CONCLUSIE: COMP-A WERKT NIET OP EEN NETWERK GETRAIND OP 1 VB VAN COMP-B
ZOALS WE AL VERWACHTTEN"""
X2,y2 = preprocessing.split_soundfile(0)
y2 = encoder.fit_transform(y2)
#%%
padded_X2 = np.zeros((len(X2), np.shape(padded_X)[1]))

for n, i in enumerate(X2):
    padded_X2[n,:len(i)] = i
    
X2 = scaler.fit_transform(np.array(padded_X2, dtype = float))

print("Evaluate on comp-a data")
results2 = model.evaluate(X2, y2, batch_size=32)
y2_pred = model.predict(X2)
print("test loss, test acc:", results2)
#%% conf mat
cm=confusion_matrix(y2,np.argmax(y2_pred,axis=1))
print(cm)