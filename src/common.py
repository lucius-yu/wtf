#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:23:52 2017

@author: eyulush
"""

import numpy as np
import pandas as pd

from multiprocessing import Pool
from ast import literal_eval


'''
papply function is to use multiprocessing for paralle processing
input
    df_list is a list of dataframes
    fxn is the function working on each dataframe in df_list
    fxn should always reture pd.Series or pd.DataFrame
return
    dataframe

example of fxn

def foo(record):
    return pd.Series({'mean' :np.mean(record.Z)})
'''

def papply(df_list, fxn, pmax=8):

    papply_pool = Pool(min(len(df_list), pmax))
    result = papply_pool.map(fxn, df_list)
    df_result = pd.concat(result)

    # clean up
    papply_pool.close()
    del papply_pool

    return df_result

'''
split the dataframe to n portations by row
'''
def split_dataframe(df_data, n=8):
    split_size = np.ceil(df_data.shape[0]/float(n))
    
    df_list=[]
    for i in range(n):
        start_index = int(i * split_size)
        end_index = int(min((i+1)*split_size, df_data.shape[0]))
        df_list.append(df_data.iloc[start_index:end_index])
    return df_list

'''
save and load data frame
example: 
    common.save_df(df_train,'../data/', 'train_data')
    common.load_df('../data/', 'train_data')
'''

def save_df(df, path, name, **kwargs ):
    df.dtypes.to_pickle(path + name + '.dtype')
    df.to_csv(path + name + '.csv.gz', compression='gzip', index=False, **kwargs)

def load_df(path, name, no_converter=False, **kwargs):
    obj_dtype = pd.read_pickle(path + name + ".dtype")

    # build up converters
    converters = dict()
    if no_converter==False:
        for k in obj_dtype.keys():
            # use getatter, when no hasobject method, we get false
            if getattr(obj_dtype[k], "hasobject",False):
                converters[k] = literal_eval
    # update int key to str key
    obj_dtype_dict = obj_dtype.to_dict()
    for k in obj_dtype_dict.keys():
        # change number key to str
        if type(k)==int:
            obj_dtype_dict[str(k)]=obj_dtype_dict.pop(k)

    # if converter in args, then update converter with parameters
    if 'converters' in kwargs.keys():
        converters.update(kwargs['converters'])

    # update parameters
    if len(converters.keys()) > 0:
        kwargs['converters'] = converters
    return pd.read_csv(path + name + '.csv.gz', compression='gzip', dtype=obj_dtype_dict, **kwargs)

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_pred - y_true) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)


class TrainingCtrl:
    def __init__(self, init_learning_rate=0.1,decay=0.99, min_learning_rate=0.05, half_min_round=500):
        self.iter = 0
        self.learning_rate=init_learning_rate
        self.min_learning_rate=min_learning_rate
        self.decay = decay
        self.half_min_round = half_min_round

    def get_learning_rate(self,iter):
        self.iter+=1
        if self.iter % self.half_min_round == 0 and self.min_learning_rate > 0.005 :
            if self.learning_rate == self.min_learning_rate:
                self.learning_rate = self.min_learning_rate/2.0
            self.min_learning_rate = self.min_learning_rate/2.0
        self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.decay )
        # print(self.learning_rate)
        return self.learning_rate

from workalendar.europe import France, Germany, UnitedKingdom, Spain
from workalendar.usa import UnitedStates
from workalendar.asia import Japan, HongKong

def get_holiday(cal):
    holiday = [str(day[0]) for day in cal.holidays(2015)] + \
              [str(day[0]) for day in cal.holidays(2016)] + \
              [str(day[0]) for day in cal.holidays(2017)]
    return holiday
            
def get_holidays():
    us_cal = UnitedStates()
    uk_cal = UnitedKingdom()
    es_cal = Spain()
    de_cal = Germany()
    fr_cal = France()
    ja_cal = Japan()
    
    us_holiday = get_holiday(us_cal)
    uk_holiday = get_holiday(uk_cal)
    de_holiday = get_holiday(de_cal)
    fr_holiday = get_holiday(fr_cal)
    
    es_holiday = get_holiday(es_cal)
    ja_holiday = get_holiday(ja_cal)
    
    ru_holiday = ['2015-11-04'] + \
                 ['2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04', '2016-01-05', '2016-01-06', '2016-01-07', \
                  '2016-02-22', '2016-02-23', '2016-03-08', '2016-05-01', '2016-05-09', '2016-06-12', '2016-06-13','2016-11-04'] + \
                 ['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07', \
                  '2017-02-23', '2017-02-24', '2017-03-08', '2017-05-01', '2017-05-08', '2017-05-09', '2017-06-12', '2017-11-04', '2017-11-06']
    zh_holiday = ['2015-09-03', '2015-09-04', '2015-09-27', '2015-10-01', '2015-10-02', '2015-10-03', '2015-10-04', '2015-10-05', '2015-10-06','2015-10-07', '2015-10-21'] + \
                 ['2016-01-01', '2016-01-02', '2016-01-03', '2016-02-07', '2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12','2016-02-13', '2016-04-04', '2016-05-01', '2016-05-02', \
                  '2016-06-09', '2016-06-10', '2016-09-15', '2016-09-16', '2016-10-03', '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07'] + \
                 ['2017-01-02', '2017-01-27', '2017-01-28', '2017-01-29', '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-02',\
                  '2017-04-02', '2017-04-03', '2017-04-04', '2017-05-01', '2017-05-28', '2017-05-29', '2017-05-30',\
                  '2017-10-01', '2017-10-02', '2017-10-03', '2017-10-04', '2017-10-05', '2017-10-06', '2017-10-07', '2017-10-08']
    zh_o_holiday = ['2015-10-10', '2016-02-06', '2016-02-14', '2016-06-12', '2016-09-18', '2016-10-08', '2016-10-09'] + \
                   ['2017-01-22', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30']

    return us_holiday, uk_holiday, de_holiday, fr_holiday, ru_holiday, es_holiday, ja_holiday, zh_holiday, zh_o_holiday
    
             