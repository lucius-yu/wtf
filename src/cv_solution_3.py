#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 20:58:57 2017

@author: eyulush
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import lightgbm as lgb
import copy

from common import parse_page
from common import get_language

from common import TrainingCtrl, smape
from common import generate_train_data, generate_real_data

train = pd.read_csv("../input/train_2.csv.zip",compression='zip')
train = train.fillna(0.)

'''
lgb training is supervised traing. So, we need x_train, y_train for training model
then x_test for predict on validation 

interesting finding, the latest training date will be more important
and 1 year ago date somehow important than other dates
'''
def solution_3(df_train, df_test):
    # x value will be fixed. only y value will be shifted one day by one day
    df_x_train = df_train.iloc[:, :-63].drop('Page',axis=1).copy()
    df_x_test = df_train.iloc[:, (63+1):].copy()
    # log transform
    x_train = np.log(df_x_train + 1.0)
    x_test = np.log(df_x_test + 1.0)
    # reset colname -- not needed.
    x_train.columns=range(x_train.shape[1])
    x_test.columns=range(x_test.shape[1])
    #
    assert(x_train.shape[1]==x_test.shape[1])
    
    s3_result = pd.DataFrame()
    for i in range(df_test.shape[1]):
        df_y_train = df_train.iloc[:, -(63-i)].copy()
        df_y_test = df_test.iloc[:,i]

        y_train = np.log(df_y_train + 1.0)
        lgb_d_train = lgb.Dataset(x_train,label=y_train.values)

        params = {'objective':'poisson',
                  'metric':'mae',
                  'learning_rate': 0.01
                  }

        tctrl = TrainingCtrl(init_learning_rate=0.1,\
                             decay=0.996,\
                             min_learning_rate=0.01)

        gbm = lgb.train(params, lgb_d_train, num_boost_round=1000, early_stopping_rounds=50, 
                        valid_sets=[lgb_d_train], learning_rates=tctrl.get_learning_rate,
                        verbose_eval=50)
    

        log_preds = gbm.predict(x_test)
        preds = np.exp(log_preds) - 1.0
        score = (smape(df_y_test.values, np.round(preds)))
        print(score)
    
        s3_result = s3_result.append(pd.DataFrame({'date': [df_y_test.name], 'score': [score]}))
    return s3_result

# the dates used for validation
valid_dates = [('2015-07-01', '2016-09-13', '2016-09-14', '2016-11-14'), 
               ('2015-07-01', '2017-03-14', '2017-03-15', '2017-05-15'),
               ('2015-07-01', '2016-10-11', '2016-10-12', '2016-12-12'), 
               ('2015-07-01', '2017-05-09', '2017-05-10', '2017-07-10')]


cv_scores = pd.read_csv('../output/cv_scores.csv') if os.path.isfile('../output/cv_scores.csv') else pd.DataFrame()
for train_start_date, train_end_date, valid_start_date, valid_end_date in valid_dates:
    # get training data
    df_train = generate_train_data(train, train_start_date, train_end_date)
    # get test data
    df_test = train[list(map(str,pd.date_range(start=valid_start_date, end=valid_end_date).strftime('%Y-%m-%d')))]
    # 
    s3_scores = solution_3(df_train, df_test)
    
    s3_scores.to_csv('../output/cv_solution_3_with_round_' + valid_start_date + '_' + valid_end_date, index=False)
    cv_scores = cv_scores.append(pd.DataFrame({'solution':['solution_3'], 'round': [True],
                                'valid_start_date': [valid_start_date], 'score': [s3_scores.mean()]}))
cv_scores.to_csv('../output/cv_scores.csv',index=False)
