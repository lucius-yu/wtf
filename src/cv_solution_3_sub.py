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
from common import generate_train_data, generate_test_data, generate_real_data
from common import get_smape_scores

'''
Control variables
'''
cross_validation = True
submission = False


# load data
train = pd.read_csv("../input/new_train_2.csv.zip",compression='zip')
train = train.fillna(0.)

'''
lgb training is supervised traing. So, we need x_train, y_train for training model
then x_test for predict on validation 

'''
def solution_3(df_train, df_test, pred_len=63, skip_len=0, save_model=False):
    test = df_test.copy()
    
    # x value will be fixed. only y value will be shifted one day by one day
    df_x_train = df_train.iloc[:, :-(pred_len)].drop('Page',axis=1).copy()
    df_x_test = df_train.iloc[:, (pred_len+1):].copy()
    # log transform
    x_train = np.log(df_x_train + 1.0)
    x_test = np.log(df_x_test + 1.0)
    
    test['date'] = test.Page.apply(lambda a: a[-10:]) # get the date
    test['Page'] = test.Page.apply(lambda a: a[:-11]) # get the page
    test_dates = np.sort(test.date.unique()).tolist()
    
    df_pred = pd.DataFrame()
    for i in range(skip_len,min(pred_len,len(test_dates))+skip_len):
        df_y_train = df_train.iloc[:, -(pred_len-i)].copy()    
        y_train = np.log(df_y_train + 1.0)
        
        lgb_d_train = lgb.Dataset(x_train,label=y_train.values, silent=True, free_raw_data=False)

        params = {'objective':'poisson',
                  'metric':'mae',
                  'num_leaves' : 127,
                  'max_depth' : 8,
                  'learning_rate': 0.01
                  }

        tctrl = TrainingCtrl(init_learning_rate=0.1,\
                             decay=0.997,\
                             min_learning_rate=0.01)

        gbm = lgb.train(params, lgb_d_train, num_boost_round=1000, early_stopping_rounds=50, 
                        valid_sets=[lgb_d_train], learning_rates=tctrl.get_learning_rate,
                        verbose_eval=50)
        if save_model:
            gbm.save_model(filename='../models/model_'+str(i) + '_' + df_y_train.name + '.mdl')
        
        preds = np.exp(gbm.predict(x_test)) - 1.0
        preds = np.round(preds)                      
        df_pred = df_pred.append(pd.DataFrame({'Page': df_train.Page.values, 'date':test_dates[i-skip_len], 'Visits' : preds}))
        if cross_validation:
            df_y_test=train[test_dates[i-skip_len]]
            print(smape(df_y_test.values, np.round(preds)))
            
    test = test.merge(df_pred, how='left', on=['Page','date'])
    return test

if cross_validation:
    # the dates used for validation
    #valid_dates = [('2015-07-01', '2016-09-13', '2016-09-14', '2016-10-13'), 
    #               ('2015-07-01', '2017-05-09', '2017-05-10', '2017-06-07')]
    valid_dates = [('2015-07-01', '2017-05-07', '2017-05-10', '2017-07-10')]

    cv_scores = pd.read_csv('../output_28_cv/cv_scores.csv') if os.path.isfile('../output_28_cv/cv_scores.csv') else pd.DataFrame()
    # train_start_date, train_end_date, valid_start_date, valid_end_date = valid_dates[0]
    for train_start_date, train_end_date, valid_start_date, valid_end_date in valid_dates:
        # get training data
        df_train = generate_train_data(train, train_start_date, train_end_date)
        # get test data
        df_test = generate_test_data(train, valid_start_date, valid_end_date)
        # get real data
        df_real = generate_real_data(train, start_date=valid_start_date, end_date=valid_end_date)
         
        s3_result = solution_3(df_train, df_test, pred_len=63, skip_len=2)
        s3_result.to_csv('../output_28_cv/s3_result_'+valid_start_date+'.csv', index=False)
        
        s3_scores = get_smape_scores(df_real, s3_result, round_flag=True)
        s3_scores.to_csv('../output_28_cv/cv_solution_3_with_round_' + valid_start_date + '_' + valid_end_date, index=False)
        print(s3_scores)
        cv_scores = cv_scores.append(pd.DataFrame({'solution':['solution_3'], 'round': [True],
                                'valid_start_date': [valid_start_date], 'score': [s3_scores.mean()]}))
        print(cv_scores)
        cv_scores.to_csv('../output_28_cv/cv_scores.csv',index=False)

if submission:
    # real solution
    real_test = pd.read_csv("../input/key_2.csv.zip",compression='zip')
    
    # if we only predict to 21 days, pred_len = 21
    # if train data only till 2017-09-10, we use skip_len = 2. i.e. skip 2 day
    s3_result = solution_3(train, real_test, pred_len=21, skip_len=2)
    s3_result['Visits'] = round(s3_result['visits'])

    s3_result.to_csv('../submit/solution_3_raw.csv',index=False)
    s3_result[['Id','Visits']].to_csv('../submit/solution_3.csv', index=False)
# load model example
# model = lgb.Booster(model_file='../models/model_'+str(i) + '_' + df_y_train.name + '.mdl')
