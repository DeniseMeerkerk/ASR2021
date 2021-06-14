#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:30:33 2021

@author: denise
"""

import numpy as np
import MFCCneuralmodel
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#%% load data

data = np.load('variablesMFCC/preprocessed_data.npz')
X_train, y_train = data['arr_0'],data['arr_3']
X_test,y_test  = data['arr_1'], data['arr_4']
X_val,y_val = data['arr_2'],data['arr_5']
del data

data = np.load('variablesMFCC/feat_classweights.npz', mmap_mode='r')
feature_dim_1, feature_dim_2, class_weights = data['arr_0'], data['arr_1'], data['arr_2']
del data

channel, epochs, batch_size, verbose, num_classes = MFCCneuralmodel.modelParameters()
#%% load model + evaluate each batch
for i in range(3):
    date_of_model = "06-02/"
    model_path = "training_"+date_of_model+str(i)+"th_batch/cp.ckpt"
    model = MFCCneuralmodel.get_model(channel,[feature_dim_1,feature_dim_2],num_classes)
    optimizer = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.binary_crossentropy,optimizer=optimizer,
          metrics=['accuracy'])
    model.load_weights(model_path)
    print("Trained model test set results" + str(i))
    results,cm = MFCCneuralmodel.evaluate(model, X_test, y_test)


