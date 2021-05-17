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
        if len(X)> 2000:
            X_train.append(X)
            y_train.append(ann[n][1])
    return X_train, y_train

def split_soundfile(file_number=0):
    wav_files, anno_files = code_snippets.getAudioFilenames()
    y_data = parse_file.read_lines(anno_files[file_number])
    X, y = splitWav(wav_files[file_number],y_data)
    return X, y

def split_all_soundfiles(part=1, segment=0):
    wav_files, anno_files = code_snippets.getAudioFilenames()
    X,y = [],[]
    filenumbers = range(int(segment*len(wav_files)*part), 
                        int(segment*len(wav_files)*part+len(wav_files)*part))
    for i in filenumbers:
        Xtemp, ytemp = split_soundfile(file_number=i)
        X.append(Xtemp)
        y.append(ytemp)
    X = [item for sublist in X for item in sublist]
    y = [item for sublist in y for item in sublist]
    
    # remove weird annotations
    weird = list(set(y))
    weird.remove('?')
    weird.remove('.')
    
    indices = [i for i, x in enumerate(y) if x in weird]
    
    for index in sorted(indices, reverse=True):
        del X[index]
        del y[index]
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

