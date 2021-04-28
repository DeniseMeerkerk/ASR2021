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


#%% split wavfile based on annotation
def splitWav(filename=code_snippets.getAudioFilenames()[0][0], ann=code_snippets.getAudioFilenames()[1][0]):
    X_train=[]
    y_train=[]
    for n,i in enumerate(ann):
        begin=ann[n][2]
        end=ann[n][3]
        X, sample_rate = librosa.load(filename, sr=None, offset=begin, duration=end-begin)
        X_train.append(X)
        y_train.append(ann[n][1])
    return X_train, y_train

def split_soundfile(file_number=0):
    wav_files, anno_files = code_snippets.getAudioFilenames()
    y_data = parse_file.read_lines(anno_files[file_number])
    X, y = splitWav(wav_files[file_number],y_data)
    return X, y

#%% quick fix to pad the data.
def pad_data(X):

    length=[]
    for i in X:
        length.append(len(i))
    padding = max(length)

    padded_X = np.zeros((len(X), padding))

    for n, i in enumerate(X):
        padded_X[n,:len(i)] = i
    return padded_X


    
    
    
    
