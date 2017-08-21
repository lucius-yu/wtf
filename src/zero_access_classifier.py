# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:23:01 2017

@author: eyulush
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# load train
train = pd.read_csv('../input/train_1.csv.zip',compression='zip')

def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res.group(0)[0:2]
    return 'na'

# add column for language
train['lang'] = train.Page.map(get_language)


# first train-test-split..
n_train = 480
df_train = train.iloc[:,0:n_train+1]
df_test = train.iloc[:, [0]+range(n_train+1,train.shape[1])]

### do statistics
def get_zero_access_rate(x):
    return float((x==0).sum())/x.notnull().sum() if x.notnull().sum() > 0 else 0.0

# zero access rate per page, the NaN will be excluded, (1673L,) pages has zero access rate > 0.5
zero_access_rate = df_train.iloc[:,1:].apply(get_zero_access_rate, axis=1)

# for different week days zero access rate per page the Nan will be excluded
def get_weekday(datetime_str):
    return pd.to_datetime(datetime_str).weekday()

week_day_zero_access_rate = df_train.iloc[:,1:].apply(lambda x: x.groupby(get_weekday).apply(get_zero_access_rate), axis=1)

# non working day is weekend + holiday
def is_nonworking_day(x, lang=None):
    #official non working days by country (manual search with google)
    #I made a lot of shortcuts considering that only Us and Uk used english idiom, 
    #only Spain for spanich, only France for french, etc
    train_us=['2015-07-04','2015-11-26','2015-12-25','2016-07-04','2016-11-24','2016-12-26']
    test_us=[]
    train_uk=['2015-12-25','2015-12-28','2016-01-01','2016-03-28','2016-05-02','2016-05-30','2016-12-26','2016-12-27']
    test_uk=['2017-01-01']
    train_de=['2015-10-03', '2015-12-25', '2015-12-26','2016-01-01', '2016-03-25', '2016-03-26', '2016-03-27', '2016-01-01', '2016-05-05', '2016-05-15', '2016-05-16', '2016-10-03', '2016-12-25', '2016-12-26']
    test_de=['2017-01-01']
    train_fr=['2015-07-14', '2015-08-15', '2015-11-01', '2015-11-11', '2015-12-25','2016-01-01','2016-03-28', '2016-05-01', '2016-05-05', '2016-05-08', '2016-05-16', '2016-07-14', '2016-08-15', '2016-11-01','2016-11-11', '2016-12-25']
    test_fr=['2017-01-01']
    train_ru=['2015-11-04','2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04', '2016-01-05', '2016-01-06', '2016-01-07', '2016-02-23', '2016-03-08', '2016-05-01', '2016-05-09', '2016-06-12', '2016-11-04']
    test_ru=['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07', '2017-02-23']
    train_es=['2015-08-15', '2015-10-12', '2015-11-01', '2015-12-06', '2015-12-08', '2015-12-25','2016-01-01', '2016-01-06', '2016-03-25', '2016-05-01', '2016-08-15', '2016-10-12', '2016-11-01', '2016-12-06', '2016-12-08', '2016-12-25']
    test_es=['2017-01-01', '2017-01-06']
    train_ja=['2015-07-20','2015-09-21', '2015-10-12', '2015-11-03', '2015-11-23', '2015-12-23','2016-01-01', '2016-01-11', '2016-02-11', '2016-03-20', '2016-04-29', '2016-05-03', '2016-05-04', '2016-05-05', '2016-07-18', '2016-08-11', '2016-09-22', '2016-10-10', '2016-11-03', '2016-11-23', '2016-12-23']
    test_ja=['2017-01-01', '2017-01-09', '2017-02-11']
    train_zh=['2015-09-27', '2015-10-01', '2015-10-02','2015-10-03','2015-10-04','2015-10-05','2015-10-06','2015-10-07','2016-01-01', '2016-01-02', '2016-01-03', '2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12', '2016-04-04', '2016-05-01', '2016-05-02', '2016-06-09', '2016-06-10', '2016-09-15', '2016-09-16', '2016-10-03', '2016-10-04','2016-10-05','2016-10-06','2016-10-07']
    test_zh=['2017-01-02', '2017-02-27', '2017-02-28', '2017-03-01']
    #in China some saturday and sundays are worked, so the following date are not not working day
    train_o_zh=['2015-10-10','2016-02-06', '2016-02-14', '2016-06-12', '2016-09-18', '2016-10-08', '2016-10-09']
    test_o_zh=['2017-01-22', '2017-02-04']
    
    holiday_dict = {'us': train_us + test_us, 
                    'uk': train_uk + test_uk,
                    'de': train_de + test_de,
                    'fr': train_fr + test_fr,
                    'ru': train_ru + test_ru,
                    'es': train_es + test_es,
                    'ja': train_ja + test_ja,
                    'zh': train_zh + test_zh,
                    'o_zh': train_o_zh + test_o_zh}
    
    if lang==None:
        lang = x.lang
    
    if x.lang=='zh':
        if x.date in holiday_dict('o_zh'):
            return False
    
    return get_weekday(x.date) >= 5 or (x.date in holiday_dict(x.lang))
    

    
# for different season zero access rate

# mean access per page

# median access 

# 


