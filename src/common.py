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