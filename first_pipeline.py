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


#%%
wav_files, anno_files, X, sample_rate = code_snippets.main() 
#%%
test = parse_file.read_lines(anno_files[2])#="fn000006.plk"
#%% split wavfile based on annotation
def splitWav(filename=wav_files[2], ann=test): #wav_files[2] =fn000006
    X_train=[]
    y_train=[]
    for n,i in enumerate(test):
        begin=test[n][2]
        end=test[n][3]
        X, sample_rate = librosa.load(filename, sr=None, offset=begin, duration=end-begin)
        X_train.append(X)
        y_train.append(test[n][1])
    return X_train, y_train

X, y = splitWav()

#%% quick fix to pad the data.
length=[]
for i in X:
    length.append(len(i))
padding = max(length)

padded_X = np.zeros((len(X), padding))

for n, i in enumerate(X):
    padded_X[n,:len(i)] = i

#%% waveplot of first (padded) segment
plt.figure(figsize=(14, 5))
librosa.display.waveplot(padded_X[0], sr=sample_rate)

#%% ANN based on: https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
encoder = LabelEncoder()
y = encoder.fit_transform(y)#Scaling the Feature columns
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
                    batch_size=128)

#%% evaluate the model
print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)
