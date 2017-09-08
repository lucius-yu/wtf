import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import lightgbm as lgb
import copy

from common import TrainingCtrl, smape

train = pd.read_csv("../input/train_2.csv.zip",compression='zip')
train = train.fillna(0.)

# common prepocessing
def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res.group(0)[0:2]
    return 'na'

# add column for language
train.insert(loc=1, column='lang',value=train.Page.map(get_language))
train['lang'] = train['lang'].astype("category")

df_train = train.iloc[:,2:]

n_total = df_train.shape[1]
n_train = 480
n_valid = 7

n_forecasts = [1,7,14,21,28,35,42,49,56]

model_collections = list()
for n_f in n_forecasts:
    f_train = range(n_train)
    x_train = np.log(df_train.iloc[:,f_train] + 1.0)
    y_train = np.log(df_train.iloc[:,f_train[-1]+n_f] + 1.0)

    f_valid = range(n_valid,n_train+n_valid)
    x_valid = np.log(df_train.iloc[:,f_valid] + 1.0)
    y_valid = np.log(df_train.iloc[:,f_valid[-1]+n_f] + 1.0)

    lgb_d_train = lgb.Dataset(x_train,label=y_train.values)
    lgb_d_valid = lgb.Dataset(x_valid,label=y_valid.values)

    params = {'objective':'poisson',
              'metric':'mae',
              'learning_rate': 0.01
              }

    tctrl = TrainingCtrl(init_learning_rate=0.1,\
                         decay=0.996,\
                         min_learning_rate=0.01)

    gbm = lgb.train(params, lgb_d_train, num_boost_round=1000, early_stopping_rounds=50, 
                    valid_sets=[lgb_d_train, lgb_d_valid], learning_rates=tctrl.get_learning_rate,
                    verbose_eval=10)
    
    model_collections.append(copy.deepcopy(gbm))
    
    log_preds = gbm.predict(x_valid)
    preds = np.exp(log_preds) - 1.0
    print(smape(df_train.iloc[:,f_valid[-1]+n_f].values, np.round(preds)))

    
