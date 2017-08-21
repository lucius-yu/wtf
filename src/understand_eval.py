# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:37:00 2017

This file is used to understand the evaluation function. 
Codes mainly from CPMP's kernel

@author: eyulush
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns

'''
Author comments:
    It is quite straightforward, the only caveat is to treat nan correctly. 
    Thanks to the official answers on the forum, we know we can use this code. 
    It handles the case where there are nan in the y_true array, but it 
    assumes there are no nan in the y_pred array.

Understanding:
    y_true is a time series which contain true value for each time point
    y_pred is a time series which you predict
    y_true could contain missing value? i.e. nan
    y_pred use 0 in case y_true value is missing
    np.nanmean : Compute the arithmetic mean along the specified axis, ignoring NaNs.
'''

# original version of smape
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_pred - y_true) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

''' 
explore case 1
    y_true is [3]
    y_pred is [x]
for different x, we can see 
    1. SMAPE is 0 when the predicted value is equal to the true value.
    2. We also see that an under estimate is penalized more than an over estimate
    3. Non-convex which odd point is at true value.
    4. consider how to construct your loss function. 
'''
y_true = np.array([3])
y_pred = np.array([1])
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)


''' 
explore case 2
    y_true is [0]
    y_pred is [x]
for different x, we can see 
    1. 
'''

y_true = np.array([0])
y_pred = np.array([0])
x = np.linspace(0,10,1000)
res = [smape(y_true, i) for i in x]
plt.plot(x, res)

'''
check the gradient and hess for smape 
conclusion :
    smape_grad and smape_hess looks correct but smape has not gradient when
    y_pred is closed to y_true.
    so approximation method is better when y_pred close to y_true
Note, the objective function will receive pred, train_data.get_label which both 
    are np.array. and returned gradient and hess are also array.
'''   


from functools import partial
import numdifftools as nd

# take the int as input
def smape(y_pred, y_true):   
    denominator = (abs(y_true) + abs(y_pred)) / 2.0
    return 0.0 if denominator==0 else np.abs(y_pred - y_true) / denominator

def approx_smape_grad_hess(y_pred, y_true):
    grads = np.concatenate([nd.Gradient(partial(smape,y_true[i]))(y_pred[i]) for i in range(y_pred.size)])
    hesss = np.concatenate([nd.Hessian(partial(smape,y_true[i]))(y_pred[i]) for i in range(y_pred.size)])
    # another np concat to transform to 1 dimension
    return np.concatenate(grads), np.concatenate(hesss) 


y_true = np.array([3, 3, 3, 0, 1, 0])
y_pred = np.array([1, 6, 2.999, 1, 0.1, 0.1])
approx_smape_grad_hess(y_pred,y_true)