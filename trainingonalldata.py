#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:33:28 2021

@author: denise
"""
#%% import packages
import MFCCneuralmodel
import preprocessing
import numpy as np


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#%%
def batchinator(X_train,y_train, n_batch):
    N = len(X_train)
    X_batches, y_batches = [],[]
    for i in range(n_batch):
        free = range(i,N,n_batch)
        print(free)
        X_batches.append(X_train[free])
        y_batches.append(y_train[free])
    return X_batches, y_batches
        

#%%
def main():
    #%% split all data and split training in different batches
    # preprocess data
    data = np.load('variablesMFCC/preprocessed_data.npz')
    X_train = data['arr_0']
    X_test = data['arr_1']
    X_val = data['arr_2']
    y_val = data['arr_5']
    y_test = data['arr_4']
    y_train = data['arr_3']
    del data
    data = np.load('variablesMFCC/feat_classweights.npz', mmap_mode='r')
    feature_dim_1, feature_dim_2, class_weights = data['arr_0'], data['arr_1'], data['arr_2']
    channel, epochs, batch_size, verbose, num_classes = MFCCneuralmodel.modelParameters()
    X_batches, y_batches = batchinator(X_train,y_train,3)
    del X_train, y_train
    
    
    model = MFCCneuralmodel.get_model(channel,[feature_dim_1,feature_dim_2],num_classes)
    optimizer = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.binary_crossentropy,optimizer=optimizer,
                  metrics=['accuracy'])
    #callbacks
    
    
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0,
                                  mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                       patience=7, verbose=1, epsilon=1e-4, 
                                       mode='min')
    
    #%% loop die loop over de verschillende train batches
    for i, (X_train,y_train) in enumerate(zip(X_batches,y_batches)):
        if i > 0:
            checkpoint_path_prev = "training_06-08/"+str(i-1)+"th_batch/cp.ckpt"
            model = MFCCneuralmodel.get_model(channel,[feature_dim_1,feature_dim_2],num_classes)
            model.compile(loss=keras.losses.binary_crossentropy,optimizer=optimizer,
                  metrics=['accuracy'])
            model.load_weights(checkpoint_path_prev)
        checkpoint_path = "training_06-08/"+str(i)+"th_batch/cp.ckpt"
        mcp_save = ModelCheckpoint(checkpoint_path, save_best_only=True,
                               save_weights_only=True, monitor='val_loss',
                               mode='min')
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
              verbose=verbose, callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
              validation_data=(X_val, y_val),
              class_weight={0:class_weights[0] , 1:class_weights[1]})
    
    
    print("Trained model test set results")
    results,cm = MFCCneuralmodel.evaluate(model, X_test, y_test)
    
    return

#%%
if __name__ == '__main__':
    main()
        


    #%% load different stages of models and evaluate them