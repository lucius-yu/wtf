# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:23:01 2017

@author: eyulush
"""

from functools import partial
import pandas as pd
import numpy as np
import common
from sklearn.metrics import confusion_matrix

langs = ['zh', 'fr', 'en', 'na', 'ru', 'de', 'ja', 'es']

### load data
page_stats = pd.read_csv('../data/page_stats.csv.gz',compression='gzip')
date_prop = pd.read_csv('../data/date_prop.csv.gz',compression='gzip')
date_lang_prop = pd.read_csv('../data/date_lang_prop.csv.gz',compression='gzip')
page_weekday_stats = pd.read_csv('../data/page_weekday_stats.csv.gz',compression='gzip')
page_non_working_day_stats = pd.read_csv('../data/page_non_working_day_stats.csv.gz',compression='gzip')

df_train = pd.read_csv('../data/df_train.csv.gz',compression='gzip')
df_train['lang'] = df_train['lang'].astype('category')
df_train['non_working_day'] = df_train['non_working_day'].astype('category')
df_train['weekday'] = df_train['weekday'].astype('category')

df_valid = pd.read_csv('../data/df_valid.csv.gz',compression='gzip')
df_valid['lang'] = df_valid['lang'].astype('category')
df_valid['non_working_day'] = df_valid['non_working_day'].astype('category')
df_valid['weekday'] = df_valid['weekday'].astype('category')

df_test = pd.read_csv('../data/df_test.csv.gz',compression='gzip')
df_test['lang'] = df_test['lang'].astype('category')
df_test['non_working_day'] = df_test['non_working_day'].astype('category')
df_test['weekday'] = df_test['weekday'].astype('category')

df_sub = pd.read_csv('../data/df_sub.csv.gz',compression='gzip')
df_sub['lang'] = df_sub['lang'].astype('category')
df_sub['non_working_day'] = df_sub['non_working_day'].astype('category')
df_sub['weekday'] = df_sub['weekday'].astype('category')


### if d_train and d_valid saved before, started from here
# d_train=pd.read_csv('../data/train_data.csv.gz',compression='gzip')
# d_valid=pd.read_csv('../data/valid_data.csv.gz', compression='gzip')

import lightgbm as lgb

f_to_use = ['lang', 'zero_access_rate', 'mean_access', 'median_access', 'std_access',\
            'dom', 'doy', 'moy', 'non_working_day', 'weekday',\
            'weekday_mean_access', 'weekday_zero_access_rate',\
            'nw_day_mean_access', 'nw_day_zero_access_rate']

lgb_d_train = lgb.Dataset(df_train[f_to_use],label=df_train.label.values)
lgb_d_valid = lgb.Dataset(df_valid[f_to_use],label=df_valid.label.values)

params = {'boosting_type': 'gbdt',
          'is_unbalance': False,
          'learning_rate': 0.01,
          'max_depth': 4,
          'metric': 'binary_logloss',
          #'metric': 'map', # mean average precision
          'min_data_in_leaf': 100,
          'min_sum_hessian_in_leaf': 20,
          'num_leaves': 16,
          'objective': 'binary',
          'scale_pos_weight': 1.0,
          'task': 'train',
          #'feature_fraction': 0.9,
          'verbose': 1}

evals_result=dict()
model = lgb.train(params, lgb_d_train,  valid_sets=lgb_d_valid, num_boost_round=2000,
                  early_stopping_rounds=100,
                  evals_result=evals_result, verbose_eval=True)

preds = model.predict(df_test[f_to_use], num_iteration=900)

# with high 
# threshold = 0.67 is better

thresholds = [0.6, 0.65, 0.67, 0.7,0.75, 0.8,0.85, 0.9, ]
for thres in thresholds:
    thres=0.5
    test_preds = preds > thres
    tn, fp, fn, tp = confusion_matrix(df_test.label.values, test_preds).ravel()
    print("tp=%d, fp=%d, gain=%d"%(tp, fp, tp-fp))
    print tp/float(fp+tp)

threshold=0.67

sub_preds = model.predict(df_sub[f_to_use], num_iteration=900)
df_sub_preds = df_sub[['Page','date']].copy()
df_sub_preds['zero_preds'] = [1 if pred > threshold else 0 for pred in sub_preds]

common.save_df(df_sub_preds,'../data/','df_sub_preds')