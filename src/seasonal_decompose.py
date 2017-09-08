#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:51:49 2017

This is debug file for finding solution 
Approach : 
    1. for each page, take the log first
    2. use seasonal_decompose to decompose trend and periodic component
    3. use linear regression for trend prediction
    4. use lstm for periodic component prediction
    5. add trend and periodic prediction and take exponential as the final prediction

@author: eyulush
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# load the data, split into test and train
train = pd.read_csv('../input/train_1.csv.zip',compression='zip')

# common prepocessing
def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res.group(0)[0:2]
    return 'na'

# add column for language
train.insert(loc=1, column='lang',value=train.Page.map(get_language))

# sample from train data
n_sample=1000
train_samples = train.sample(n_sample)

CELL_SIZE=20
TIME_STEPS=183
INPUT_SIZE=1
PRED_STEPS=20
LR=0.001
EPOCHS=50

def get_batch(series):
    global TIME_STEPS
    
    TOTAL_STEPS = TIME_STEPS + PRED_STEPS
    
    train_seq = np.array([series.iloc[idx:idx+TOTAL_STEPS].tolist() for idx in range(series.shape[0]-TOTAL_STEPS+1)])
   
    x_seq = train_seq[:,:TIME_STEPS]
    y_seq = train_seq[:,TIME_STEPS:]
    
    return [x_seq[:, :, np.newaxis], y_seq]
 
def decompose_processing(ts,freq):
    ts_decomp = seasonal_decompose(ts,freq=freq,model="additive")
    
    # print information
    print(abs(ts_decomp.seasonal).mean())
    print(abs(ts_decomp.resid).mean())
    print(abs(ts_decomp.trend).mean())
    
    # plot information
    plt.subplot(411)
    plt.plot(ts, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(ts_decomp.trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(ts_decomp.seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(ts_decomp.resid, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    
    return ts_decomp
    
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Perform Dickey-Fuller test:
    #print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #print(dfoutput)
    return dftest[1]

search_idx = [-3, -2,-1,0,1,2,3]  
search_result = list()
i=2
i=206
i=9

for i in range(300):
    series = train_samples.iloc[i,2:]
    # drop the initial nan
    series = series[series.notnull().cumsum()>0]
    series = series.fillna(0.0)
    series.index=pd.to_datetime(series.index)
    series = series.astype('float32')
    train_ts = series[:-PRED_STEPS]
    test_ts = series[-PRED_STEPS:]
    
    y_true = train_ts[-60:]
    result = list()
    
    if y_true.sum() > 49 and y_true.median() > 0.0:
        for j in search_idx:
            y_pred = y_true.sort_values()[24+j]
            if y_pred == y_true.median() and j != 0:
                result.append(10000.0)
            else:
                result.append(smape_sv(y_pred,y_true))   
        if np.argmin(result) == 6:
            print(i)
        search_result.append(search_idx[np.argmin(result)])


    print(smape_sv(y_true.mean(),y_true))
    print(smape_sv(y_true.median(),y_true))
    
    print(smape_sv(y_true.mean(),test_ts))
    print(smape_sv(y_true.median(),test_ts))
    
    test_ts = series[-PRED_STEPS:]
    train_ts[-49:].plot()
    test_ts.plot()
    if train_ts.shape[0] > 50 and (train_ts>0).sum()>50:
        if (test_stationarity(train_ts) > 0.5):
            print(i)
            print(test_stationarity(train_ts))

# for each sample
for i in range(n_sample):
    series = train_samples.iloc[i,2:]
    # drop the initial nan
    series = series[series.notnull().cumsum()>0]
    series = series.fillna(0.0)
    series.index=pd.to_datetime(series.index)
    series = series.astype('float32')
    plt.plot(series)
    # train and test timeseries
    train_ts = series[:-PRED_STEPS]
    test_ts = series[-PRED_STEPS:]
    # take the log
    train_ts_log = np.log(train_ts+1)
    # seperate the weekdays
    xx=train_ts_log[train_ts.index.weekday==2]
    xx_decomp=decompose_processing(xx,36)
    
    test_stationarity(train_ts)
    
    # decompose to trend and periodic
    decomposition = decompose_processing(np.log(series),240)
    
    # trend, seasonal
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    resid = decomposition.resid
    
    #if abs(seasonal).mean() > abs(resid).mean():
    print(abs(seasonal).mean())
    print(abs(resid).mean())
    print(abs(trend).mean())

    '''    
    plt.plot(trend)
    plt.plot(seasonal)
    plt.plot(resid)
    '''
    
    # build model
    model = Sequential()
    # build a LSTM RNN
    model.add(LSTM(units=CELL_SIZE, 
                   input_shape=(TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
                   return_sequences=False,      # True: output at all steps. False: output as last step.
                   #stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
                   ))
    model.add(Dense(PRED_STEPS))
    adam = Adam(LR)
    model.compile(optimizer=adam,loss='mse',)
    
    x_train, y_train = get_batch(seasonal)
    x_test = (seasonal[-TIME_STEPS:].values).reshape(1,TIME_STEPS,1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # fit the model
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=1, verbose=2, validation_split=0.2, callbacks=[early_stopping])