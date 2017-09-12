'''
Try with 21 and 28 days model

'''
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

# train = pd.read_csv("../input/train_2.csv.zip",compression='zip')
train = pd.read_csv("../input/new_train_2.csv.zip",compression='zip')

train = train.fillna(0.)

valid_dates = [('2015-07-01', '2016-09-13', '2016-09-14', '2016-11-14'), 
               ('2015-07-01', '2017-03-14', '2017-03-15', '2017-05-15'),
               ('2015-09-01', '2016-10-11', '2016-10-12', '2016-12-12'), 
               ('2015-07-01', '2017-05-09', '2017-05-10', '2017-07-10')]


def solution_3(df_train, df_test):
    # x value will be fixed. only y value will be shifted one day by one day
    df_x_train = df_train.iloc[:, :-21].drop('Page',axis=1).copy()
    df_x_test = df_train.iloc[:, (21+1):].copy()
    # log transform
    x_train = np.log(df_x_train + 1.0)
    x_test = np.log(df_x_test + 1.0)
    
    # reset colname
    x_train.columns=range(x_train.shape[1])
    x_test.columns=range(x_test.shape[1])
    #
    assert(x_train.shape[1]==x_test.shape[1])
    
    df_y_train = df_train.iloc[:, -1].copy()
    df_y_test = df_test.iloc[:,20].copy()
    
    y_train = np.log(df_y_train + 1.0)
    y_test = np.log(df_y_test + 1.0)

    lgb_d_train = lgb.Dataset(x_train,label=y_train.values)
    lgb_d_test = lgb.Dataset(x_test,label=y_test.values)
    
    params = {'objective':'poisson',
              'metric':'mae',
              'num_leaves' : 64,
              'max_depth' : 8,
              'learning_rate': 0.01
             }

    tctrl = TrainingCtrl(init_learning_rate=0.1,\
                     decay=0.997,\
                     min_learning_rate=0.01)

    gbm = lgb.train(params, lgb_d_train, num_boost_round=2000, early_stopping_rounds=50, 
                valid_sets=[lgb_d_train, lgb_d_test], learning_rates=tctrl.get_learning_rate,
                verbose_eval=10)
    
    print("Features importance...")
    gain = gbm.feature_importance('gain')
    ft = pd.DataFrame({'feature':gbm.feature_name(), 'split':gbm.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    print(ft)
    preds = np.exp(gbm.predict(x_test)) - 1.0
    print(smape(df_y_test.values, np.round(preds)))


train_start_date, train_end_date, valid_start_date, valid_end_date = valid_dates[3]
df_train = generate_train_data(train, train_start_date, train_end_date)
df_test = train[list(map(str,pd.date_range(start=valid_start_date, end=valid_end_date).strftime('%Y-%m-%d')))]


