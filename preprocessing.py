#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 2021

@author: denise, nienke
"""

import librosa
import warnings
warnings.filterwarnings('ignore')
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import code_snippets
import parse_file

def split_soundfile():
    wav_files, anno_files, X, sample_rate = code_snippets.main() 
    test = parse_file.read_lines(anno_files[2])#="fn000006.plk"
    #%% split wavfile based on annotation
    def splitWav(filename=wav_files[2], ann=test): #wav_files[2] =fn000006
        X_train=[]
        y_train=[]
        for n,i in enumerate(ann):
            begin=ann[n][2]
            end=ann[n][3]
            X, sample_rate = librosa.load(filename, sr=None, offset=begin, duration=end-begin)
            X_train.append(X)
            y_train.append(ann[n][1])
        return X_train, y_train
    #%%
    X, y = splitWav()
    return X, y, sample_rate

def pad_data(X):
    #%% quick fix to pad the data.
    length=[]
    for i in X:
        length.append(len(i))
    padding = max(length)

    padded_X = np.zeros((len(X), padding))

    for n, i in enumerate(X):
        padded_X[n,:len(i)] = i
    return padded_X