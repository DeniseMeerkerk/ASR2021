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

def split_soundfile(file_number=0,wav_files=code_snippets.getAudioFilenames()[0],
                    anno_files=code_snippets.getAudioFilenames()[1]):
    y_data = parse_file.read_lines(anno_files[file_number])
    X, y = splitWav(wav_files[file_number],y_data)
    return X, y

def split_all_soundfiles(part=1, segment=0):
    wav_files, anno_files = code_snippets.getAudioFilenames()
    X,y = [],[]
    filenumbers = range(int(segment*len(wav_files)*part), 
                        min(int(segment*len(wav_files)*part+len(wav_files)*part),len(wav_files)))
    for i in filenumbers:
        Xtemp, ytemp = split_soundfile(file_number=i,wav_files=wav_files,
                                       anno_files=anno_files)
        X.append(Xtemp)
        y.append(ytemp)
    X = [item for sublist in X for item in sublist]
    y = [item for sublist in y for item in sublist]
    
    # remove weird annotations
    weird = list(set(y))
    if '?' in weird:
        weird.remove('?')
    if '.' in weird:
        weird.remove('.')
    
    indices = [i for i, x in enumerate(y) if x in weird]
    
    for index in sorted(indices, reverse=True):
        del X[index]
        del y[index]
    return X, y
        


def train_test_val_split(train_part=0.6, test_val_part=0.2):
    train_segment=(1-train_part)/train_part/2
    X_train, y_train = split_all_soundfiles(part=train_part, segment=train_segment)
    #print('train complete')
    test_segment1=0
    test_segment2=(2/test_val_part)-1
    X_test1, y_test1 = split_all_soundfiles(part=test_val_part, segment=test_segment1)
    X_test2, y_test2 = split_all_soundfiles(part=test_val_part, segment=test_segment2)
    X_test = X_test1 + X_test2
    y_test = y_test1 + y_test2
    #print('test complete')
    X_val1, y_val1 = split_all_soundfiles(part=test_val_part, segment=test_segment1+1)
    X_val2, y_val2 = split_all_soundfiles(part=test_val_part, segment=test_segment2-1)
    X_val = X_val1 + X_val2
    y_val = y_val1 + y_val2
    #print('val complete')
    return X_train, y_train, X_test, y_test, X_val, y_val
#X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split()
    
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

