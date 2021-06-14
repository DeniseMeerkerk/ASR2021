#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:29:59 2021

@author: denise
"""

#%%Code snippets (python)
# Read wav and sample frequency:
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def getAudioFilenames():
    '''
    Returns
    -------
    wav_files : list
        DESCRIPTION: returns a list of audio (.wav) files that are found 
        relatively from the current file location.
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = dir_path.replace("ASR2021",'Data/CGN_comp_ab/vol/bigdata2/corpora2/CGN2/data/audio/wav')
    

    wav_files =[]  
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith(".wav"):
                wav_files.append(os.path.join(root,name))
    wav_files.sort()
    
    anno_dir = dir_path.replace("ASR2021",'Data/CGN_comp_ab/vol/bigdata2/corpora2/CGN2/data/annot/text/plk')
    anno_files =[]
        
    for root, dirs, files in os.walk(anno_dir):
        for name in files:
            if name.endswith(".plk"):
                anno_files.append(os.path.join(root,name))
    anno_files.sort()
    
    return wav_files, anno_files
#wav_files, anno_files = getAudioFilenames()
#%% Create Mel frequency spectrogram

def plotMFS(filename):
    '''
    Parameters
    ----------
    filename : string
        DESCRIPTION: given a filename(from getAudioFilenames) of audiofile a 
        Mel frequency spectrogram is plotted

    Returns
    X : Float 32 array
        DESCRIPTION: audio time series
    sample_rate : int
        DESCRIPTION: sampling rate of X
    -------
    None.

    '''
    
    X, sample_rate = librosa.load(filename, sr=None, offset=0)#wav_files[0]
    #Create spectrogram:
    spect = librosa.feature.melspectrogram(y=X, sr=sample_rate,
                                           hop_length=int(sample_rate/100))
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(spect, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sample_rate,
                             fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    ax.set(title='Mel-frequency spectrogram')
    return X, sample_rate
#%% Create MFCC matrix:
# https://librosa.org/doc/main/generated/librosa.feature.mfcc.html

def plotMFCC(X, sample_rate):
    '''
    Parameters
    ----------
    X : Float 32 array
        DESCRIPTION: audio time series
    sample_rate : int
        DESCRIPTION: sampling rate of X

    Returns
    -------
    None.

    '''
    mfcc2 = librosa.feature.mfcc(y=X, sr=sample_rate, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0)
    fig, ax = plt.subplots()
    im = librosa.display.specshow(mfcc2, x_axis='time')
    #im.title("MFCC")
    fig.colorbar(im, ax=ax)
    
    
    #mfcc3 = librosa.feature.mfcc(y=X, sr=sample_rate, S=None, n_mfcc=20, dct_type=3, norm='ortho', lifter=0)
    #fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    #img2 = librosa.display.specshow(mfcc2, x_axis='time', ax=ax[0])
    #ax[0].set(title='Title: MFCC (dct_type=2)')
    #fig.colorbar(img2, ax=[ax[0]])
    #img3 = librosa.display.specshow(mfcc3, x_axis='time', ax=ax[2])
    #ax[2].set(title='HTK-style (dct_type=3)')
    #fig.colorbar(img3, ax=[ax[2]])
    #ax[1].remove()
    return

#%%
def main():
    wav_files, anno_files = getAudioFilenames()
    X, sample_rate = plotMFS(wav_files[2])
    plotMFCC(X, sample_rate)
    return wav_files, anno_files, X, sample_rate
#%%
if __name__ == '__main__':
    wav_files, anno_files, X, sample_rate = main()

# https://github.com/wilrop/Import-CGN

'''
fig, ax = plt.subplots()
im = librosa.display.specshow(X_train[5367].reshape((20,-1)), x_axis='time')
fig.colorbar(im, ax=ax)'''

