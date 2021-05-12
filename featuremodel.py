# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 2021

@author: denise, nienke
"""

import numpy as np

import code_snippets
import parse_file
import preprocessing

from sklearn.neighbors import KNeighborsClassifier
import parselmouth

#Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # Use seaborn's default style to make attractive graphs

# Code
def feature_extraction(X):
    features = []
    
    for i,x in enumerate(X):
        features_for_x = []
        # Get maximum volume (amplitude) in the last ten percent of utterence
        ten_perc = int(0.1*len(x))
        features_for_x.append(np.max(x[:-ten_perc])) # max volume of last ten percent of the file
        features_for_x.append(np.max(x[:-ten_perc]) - np.max(x[ten_perc:])) # difference in max volume between last ten percent and first ten percent of sentence
        
        # Use parselmouth to extract more interesting features
        snd = parselmouth.Sound(x)
        features_for_x.append(snd.get_energy())
        features_for_x.append(snd.get_intensity())
        
        #print(x)
        print(i, len(x))
        pitch = snd.to_pitch()
        print(pitch.quantile)
        features_for_x.append(pitch.get_value_in_frame(0))
        
    
    
        features.append(features_for_x)
    return np.matrix(features)

def run_knn(features, y):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(features, y)
    print(neigh.score(features,y))

# The following three plotting functions have been taken from: https://github.com/YannickJadoul/Parselmouth
def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

# This function tests the parselmouth library on the first sound file in our data
def test_parselmouth(X):
    # Plot nice figures using Python's "standard" matplotlib library
    snd = parselmouth.Sound(X[0])
    plt.figure()
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")
    
    intensity = snd.to_intensity()
    spectrogram = snd.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_intensity(intensity)
    plt.xlim([snd.xmin, snd.xmax])
    plt.show() # or plt.savefig("spectrogram.pdf")
    
    pitch = snd.to_pitch()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pre_emphasized_snd = snd.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_pitch(pitch)
    plt.xlim([snd.xmin, snd.xmax])
    plt.show() # or plt.savefig("spectrogram_0.03.pdf")

X,y = preprocessing.split_soundfile()
features = feature_extraction(X)
print(features)
run_knn(features, y)




