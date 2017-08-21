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
train.insert(loc=1, column='lang',value=train.Page.map(get_language))

# not date columns
non_date_cols = ['Page','lang']

### first train-test-split.
### prior is used to do stats and training, valid is used to validation
n_prior = 430
n_valid = 60
n_test = 60
df_prior = train.iloc[:,0:n_prior+2]
df_valid = train.iloc[:, [0,1]+range(n_prior+2, n_prior+n_valid)]
df_test = train.iloc[:, [0,1]+range(n_prior+n_valid+2,n_prior+n_valid+n_test)]

### do statistics
### page related properities
page_stats_keys = ['Page', 'lang', 'zero_access_rate', 'mean_access', 'median_access', 'std_access']
# identfy zero access day / total avail days, the number higher --> high probability of 0 access on that day
def get_zero_access_rate(x):
    return float((x==0).sum())/x.notnull().sum() if x.notnull().sum() > 0 else 0.0

# zero access rate per page, the NaN will be excluded, (1673L,) pages has zero access rate > 0.5
debug = True
if debug:
    debug_n = 21
    df_prior = df_prior.iloc[:debug_n,:]

zero_access_rate = df_prior.iloc[:,2:].apply(get_zero_access_rate, axis=1).rename('zero_access_rate')
mean_access = df_prior.iloc[:,2:].apply(lambda x: x.mean(),axis=1).rename('mean_access')
median_access = df_prior.iloc[:,2:].apply(lambda x: x.median(),axis=1).rename('median_access')
std_access = df_prior.iloc[:,2:].apply(lambda x: x.std(),axis=1).rename('std_access')

page_stats = pd.concat([df_prior[['Page','lang']],  zero_access_rate, mean_access, median_access, std_access],axis=1)

### date related properities, day of week, day of month, month of year, day of year
date_keys = ['dow', 'dom', 'moy', 'doy']

def get_dow(datetime_str):
    return pd.to_datetime(datetime_str).weekday()

def get_dom(datetime_str):
    return pd.to_datetime(datetime_str).day

def get_doy(datetime_str):
    return pd.to_datetime(datetime_str).dayofyear

date_prop = pd.DataFrame({'date':train.keys()[2:], 
                          'dow': train.keys()[2:].map(get_dow),
                          'dom': train.keys()[2:].map(get_dom),
                          'doy': train.keys()[2:].map(get_doy)})

### lang and langXdate properties
date_lang_keys = ['date','lang','non_working_day']
date_lang_prop = pd.DataFrame(columns=['date','lang'])

for lang in train.lang.unique():
    dlp = pd.DataFrame(date_prop['date'])
    dlp['lang'] = pd.Series([lang]*dlp.shape[0])
    date_lang_prop = date_lang_prop.append(dlp,ignore_index=True)    

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
    
    holiday_dict = {'en': train_us + train_uk + test_us + test_uk,
                    'de': train_de + test_de,
                    'fr': train_fr + test_fr,
                    'ru': train_ru + test_ru,
                    'es': train_es + test_es,
                    'ja': train_ja + test_ja,
                    'zh': train_zh + test_zh,
                    'na': [],
                    'o_zh': train_o_zh + test_o_zh}
    
    if lang==None:
        lang = x.lang
    if x.lang=='zh':
        if x.date in holiday_dict['o_zh']:
            return False
    return get_dow(x.date) >= 5 or (x.date in holiday_dict[x.lang])

date_lang_prop['non_working_day'] = date_lang_prop.apply(is_nonworking_day,axis=1)

# setup training data
# valid set start as df_valid.keys()[2], 2016-09-03
train_start_index = np.where(df_prior.keys()=='2015-09-03')[0][0]
df_train = df_prior.iloc[:, [0,1] + range(train_start_index, train_start_index+n_valid)]

# flatten the train data for binary classification
d_train = pd.DataFrame()
for index, row in df_train.iloc[:20,:].iterrows():
    d_train=d_train.append(pd.DataFrame([{'Page': row.Page, 'date': date, 'label' : row[date]==0} for date in row.keys()[2:]]))

# merge the properties
d_train = d_train.merge(page_stats,how='left',on='Page')
d_train = d_train.merge(date_prop, how='left',on='date')


d_valid = pd.DataFrame()
for index, row in df_valid.iloc[:20,:].iterrows():
    d_valid=d_valid.append(pd.DataFrame([{'Page': row.Page, 'date': date, 'label' : row[date]==0} for date in row.keys()[2:]]))
d_valid = d_valid.merge(page_stats,how='left',on='Page')
d_valid = d_valid.merge(date_prop, how='left',on='date')

import lightgbm as lgb

d_train = lgb.Dataset(d_train.iloc[:,3:],label=d_train.label.values)
d_valid = lgb.Dataset(d_valid.iloc[:,3:],label=d_valid.label.values)

params = {'boosting_type': 'gbdt',
          'is_unbalance': False,
          'learning_rate': 0.01,
          'max_depth': 4,
          'metric': 'binary_logloss',
          'min_data_in_leaf': 100,
          'min_sum_hessian_in_leaf': 20,
          'num_leaves': 16,
          'objective': 'binary',
          'scale_pos_weight': 1.0,
          'task': 'train',
          #'feature_fraction': 0.9,
          'verbose': 1}

cv_result = lgb.cv(params, d_train, num_boost_round=200, nfold=5, stratified=True, \
                           shuffle=True, init_model=None, feature_name='auto', \
                           categorical_feature='auto', early_stopping_rounds=300, \
                           fpreproc=None, verbose_eval=True, show_stdv=True, seed=1234, callbacks=None)

print cv_result
