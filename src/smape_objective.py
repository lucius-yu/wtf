# -*- coding: utf-8 -*-
"""
This script defined smape cost function and correspond grad and hess (1-dimension)

@author: eyulush
"""
import numpy as np
import numdifftools as nd

from functools import partial

# take the int as input
def smape(y_pred, y_true):   
    denominator = (abs(y_true) + abs(y_pred)) / 2.0
    return 0.0 if denominator==0 else np.abs(y_pred - y_true) / denominator

def approx_smape_grad_hess(y_pred, y_true):
    grads = np.concatenate([nd.Gradient(partial(smape,y_true[i]))(y_pred[i]) for i in range(y_pred.size)])
    hesss = np.concatenate([nd.Hessian(partial(smape,y_true[i]))(y_pred[i]) for i in range(y_pred.size)])
    # another np concat to transform to 1 dimension
    return np.concatenate(grads), np.concatenate(hesss) 

# define objective for xgboost or lightgbm
def smape_objective(preds,train_data):
    grad, hess = approx_smape_grad_hess(preds,train_data.get_label())
    return grad, hess

def smape_error(preds, train_data):
    values = zip(preds,train_data.get_label())
    result = np.mean([smape(pred,true) for pred, true in values])
    return 'smape_error', result, False

if __name__ == "__main__":
    y_true = np.array([3, 3, 3, 0, 1, 0])
    y_pred = np.array([1, 6, 2.999, 1, 0.1, 0.1])
    print approx_smape_grad_hess(y_pred,y_true)
    
    # self test with lgm
    import lightgbm as lgb
    import pandas as pd
    # load or create your dataset
    print('Load data...')
    df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
    df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
    
    y_train = df_train[0].values
    y_test = df_test[0].values
    X_train = df_train.drop(0, axis=1).values
    X_test = df_test.drop(0, axis=1).values

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'auc'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
    }