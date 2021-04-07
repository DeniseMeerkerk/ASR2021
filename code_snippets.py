#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:29:59 2021

@author: denise
"""

#%%Code snippets (python)
# Read wav and sample frequency:
import librosa
X, sample_rate = librosa.load(<file name>, sr=None, offset=0)
#Create spectrogram:
spect = librosa.feature.melspectrogram(y=X, sr=sample_rate, 
hop_length=sample_rate/100)
#Create MFCC matrix:
#https://librosa.org/doc/main/generated/librosa.feature.mfcc.html