#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:42:35 2017


lstm 20days model

@author: eyulush
"""

import pandas as pd
import numpy as np
import re
import gc; gc.enable()
import matplotlib.pyplot as plt

import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import text
from sklearn import naive_bayes

def symemetric_mean_absolute_percentage_error(y_true, y_pred):
    from keras import backend as K
    diff = K.abs(y_true - y_pred) / K.clip(K.abs(y_true) + K.abs(y_pred) ,
                                            K.epsilon(),
                                            None)
    return 200. * K.mean(diff, axis=-1)

def smape(y_pred, y_true):
    diff = np.abs(y_true-y_pred) / (np.abs(y_true) + np.abs(y_pred))
    return 200. * np.mean(diff)

def min_max_transform(window_data):
    scaled_data = []
    scales = []
    for window in window_data:
        x_min = np.min(window)
        x_max = np.max(window)
        normalised_window = (window - x_min) / (x_max - x_min)
        scaled_data.append(normalised_window)
        scales.append([x_min, x_max])
    return scaled_data, scales

# inv transform
# x*(x_max-x_min) + x_min
def min_max_inv_transforms(scaled_data, scales):
    
    assert(scaled_data.shape[0] == len(scales))
    return [scaled_data[i] * (scales[i][1] - scales[i][0]) + scales[i][0] \
            for i in range(scaled_data.shape[0])]

# seq_data : np.array
def process_seq_data(seq_data, seq_len=100, pred_len=20, train_split=370, normalize_window=True):
    # slide the data
    window_len = seq_len + pred_len
    result = np.array([seq_data[index: index + window_len] for index in range(\
              len(seq_data) - window_len)] ) 
    
    # min max scale the data
    if normalize_window:
        result, scales = min_max_transform(result)
        result = np.array(result)
    else:
        n = result.shape[0]
        scales=np.concatenate([np.zeros(n).reshape(-1,1),np.ones(n).reshape(-1,1)],axis=1)

    # split to train and test
    train = result[:train_split, :]
    np.random.shuffle(train)

    x_train = train[:, :-pred_len]
    y_train = train[:, -pred_len:] # the last point as label
    x_test = result[train_split:, :-pred_len]
    y_test = result[train_split:, -pred_len:]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  
    
    return [x_train, y_train, x_test, y_test, scales]

# define lstm model
def build_model(layers, learning_rate=0.01):
    model = Sequential()

    model.add(LSTM(
        #activation='relu',
        input_shape=(layers[1], layers[0]),
        units=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        # activation='relu',
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        units=layers[3]))
    model.add(Activation("linear"))

    # model.compile(loss=s_mean_absolute_percentage_error, optimizer=keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0))
    model.compile(loss=symemetric_mean_absolute_percentage_error, optimizer=keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0))
    return model


# support function
def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res.group(0)[0:2]
    return 'na'

train = pd.read_csv("../input/train_1.csv.zip", compression='zip')

# process for language
langs = ['zh', 'fr', 'en', 'na', 'ru', 'de', 'ja', 'es']

# add column for language
train.insert(loc=1, column='lang',value=train.Page.map(get_language))

# sampling part of data 
n_sample=5000
train_samples = train.sample(n_sample)

tot_pred_result = list()
tot_median_result = list()
for i in range(10):
    # get first data
    data = train_samples.iloc[i,2:].copy()
    # drop the initial nan
    data = data[data.notnull().cumsum()>0]
    
    data.index = pd.to_datetime(data.index)
    data = data.astype(float).fillna(0.0)
   
    week_data=data.groupby([(data.index.year),(data.index.week)]).median()
    #week_data = week_data.iloc[1:-1]

    seq_data = week_data.values
    seq_len=30
    pred_len = 1
               
    x_train, y_train, x_test, y_test, scales = process_seq_data(seq_data, seq_len, pred_len, train_split=30, normalize_window=False)

    lstm_model = build_model([1, seq_len, 50, pred_len],np.median(seq_data[:(seq_data.shape[0]-x_test.shape[0])])/1000.0)
    epochs=25
    lstm_model.fit(
                   x_train,
                   y_train,
                   #batch_size=128,
                   epochs=epochs,
                   validation_split=0.1)

    predicted = lstm_model.predict(x_test)
    
    # real data
    indexed_data = data.groupby([(data.index.year),(data.index.week)]).apply(lambda x: x.values)
    ds_real_data = indexed_data.loc[week_data.iloc[-predicted.shape[0]:].index]
    ds_real_data.reset_index(inplace=True, drop=True)
    pred_result = list()
    median_result = list()
    median = np.median(seq_data[:(seq_data.shape[0]-x_test.shape[0])])
    for index, frame in ds_real_data.iteritems():
        if frame.shape[0]==7:
            pred_result.append(smape(predicted[index]*np.ones(7), frame))
            median_result.append(smape(median*np.ones(7),frame))
    #print(pred_result)
    if np.mean(pred_result) != np.nan:
        print(np.mean(pred_result))
        tot_pred_result.append(np.mean(pred_result))
        #print(median_result)
        print(np.mean(median_result))
        tot_median_result.append(np.mean(median_result))
print(np.mean(tot_pred_result))    
print(np.mean(tot_median_result))    
    
tot_pred_result = list()
tot_median_result = list()
for i in range(100):
    data = train_samples.iloc[i].copy()
    data.fillna('0', inplace=True)
    seq_data = data[2:].astype(float).values
               
    seq_len=60
    pred_len = 20
               
    x_train, y_train, x_test, y_test, scales = process_seq_data(seq_data, seq_len, pred_len, normalize_window=False)
    lstm_model = build_model([1, seq_len, 100, pred_len],np.median(seq_data[:370])/1000)

    epochs=3
    lstm_model.fit(
	    x_train,
	    y_train,
	    #batch_size=128,
	    epochs=epochs,
	    validation_split=0.1)

    predicted = lstm_model.predict(x_test)

    pred_data = min_max_inv_transforms(predicted, scales[-predicted.shape[0]:])
    real_data = min_max_inv_transforms(y_test,scales[-y_test.shape[0]:])


    smape_pred_result = [smape(pred_data[i], real_data[i]) for i in range(len(pred_data))]
    print(np.mean(smape_pred_result))
    tot_pred_result.append(np.mean(smape_pred_result))
    
    smape_median_result = [smape(np.median(seq_data[:370]) * np.ones(pred_len), real_data[i])for i in range(len(real_data))]
    print(np.mean(smape_median_result))
    tot_median_result.append(np.mean(smape_median_result))

tot_pred_result=np.nan_to_num(tot_pred_result)
tot_median_result = np.nan_to_num(tot_median_result)
print(np.mean(tot_pred_result))
print(np.mean(tot_median_result))
# the first 100 samples for clustering

# pca 
from sklearn.decomposition import PCA
data = train_samples.iloc[:100,2:].fillna(0).values
pca_data = PCA(n_components=2).fit_transform(data)

plt.scatter(pca_data[:,0], pca_data[:,1])

lstm_bst=np.array(tot_pred_result)<np.array(tot_median_result)
median_bst = np.array(tot_pred_result)>np.array(tot_median_result)
plt.scatter(pca_data[lstm_bst,0], pca_data[lstm_bst,1], color='g')
plt.scatter(pca_data[median_bst,0], pca_data[median_bst,1], color='r')

gain=tot_median_result-tot_pred_result
k=5
gain_idx=gain.argsort()[-k:][::-1]
loss_idx=gain.argsort()[:k]

idx = -1

idx = idx+1
train_samples.iloc[gain_idx[idx],2:].plot()

idx = -1

idx = idx+1
train_samples.iloc[loss_idx[idx],2:].plot()

pd.qcut(train_samples.iloc[75,2:], [0, .95, 1.])

from scipy.fftpack import dct, fft
s75_dct = fft(train_samples.iloc[75,2:].values)
plt.plot(s75_dct)


s92_dct = fft(train_samples.iloc[92,2:].fillna(0).values)
plt.plot(s92_dct)
