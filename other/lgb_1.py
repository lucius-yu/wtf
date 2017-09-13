import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import lightgbm as lgb

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

#
debug = False
if debug:
    n_sample = 10000
    train = train.sample(n_sample)

n_valid = 10

f_train = list(range(2,train.shape[1]-2-n_valid))
f_valid = list(range(3,train.shape[1]-1-n_valid))

x_train = np.log(train.iloc[:,f_train] + 1.0)
y_train = np.log(train.iloc[:,f_train[-1]+1] + 1.0)

x_valid = np.log(train.iloc[:,f_valid] + 1.0)
y_valid = np.log(train.iloc[:,f_valid[-1]+1] + 1.0)

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

log_preds = gbm.predict(x_valid)
preds = np.exp(log_preds) - 1.0
print(smape(train.iloc[:,f_valid[-1]+1].values, np.round(preds)))

# update x_valid
x_valid['pred_1'] = log_preds
x_valid = x_valid.iloc[:,1:]

log_preds = gbm.predict(x_valid)
preds = np.exp(log_preds) - 1.0
print(smape(train.iloc[:,f_valid[-1]+2].values, np.round(preds)))


x_valid['pred_2'] = log_preds
x_valid = x_valid.iloc[:,1:]

log_preds = gbm.predict(x_valid)
preds = np.exp(log_preds) - 1.0
print(smape(train.iloc[:,f_valid[-1]+3].values, np.round(preds)))


x_valid['pred_4'] = log_preds
x_valid = x_valid.iloc[:,1:]

log_preds = gbm.predict(x_valid)
preds = np.exp(log_preds) - 1.0
print(smape(train.iloc[:,f_valid[-1]+4].values, np.round(preds)))


x_valid['pred_5'] = log_preds
x_valid = x_valid.iloc[:,1:]

log_preds = gbm.predict(x_valid)
preds = np.exp(log_preds) - 1.0
print(smape(train.iloc[:,f_valid[-1]+5].values, np.round(preds)))

x_valid['pred_7'] = log_preds
x_valid = x_valid.iloc[:,1:]

log_preds = gbm.predict(x_valid)
preds = np.exp(log_preds) - 1.0
print(smape(train.iloc[:,f_valid[-1]+7].values, np.round(preds)))



'''
forecast 1 day
param : with lang, poisson, mae, 0.1, 0.01, 0.996 decay
result : 37.6737

param : without lang, poisson, mae, 0.1, 0.01, 0.996 decay
result : 37.6337

forecast 10 days

param : without lang, poisson, mae, 0.1, 0.01, 0.996 decay
result : 46.2997

'''