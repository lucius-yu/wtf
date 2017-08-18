# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:17:03 2017

@author: eyulush
"""

import numpy as np

from scipy.optimize import fmin_cg

# fmin_cg example
args = (2, 3, 7, 8, 9, 10)

def f(x, *args):
    u, v = x
    a, b, c, d, e, f = args
    return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f

def gradf(x, *args):
    u, v = x
    a, b, c, d, e, f = args
    gu = 2*a*u + b*v + d     # u-component of the gradient
    gv = b*u + 2*c*v + e     # v-component of the gradient
    return np.asarray((gu, gv)) # 2 dimension of gradient

x0 = np.asarray((0, 0))
res1 = fmin_cg(f, x0, fprime=gradf, args=args)


# smape
# input, y_pred as np array
#        y_true as np.array
def smape(y_pred, y_true):   
    y_pred = np.array([y_pred]) if type(y_pred) == int else y_pred
    y_true = np.array([y_true]) if type(y_true) == int else y_true    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true) / denominator
    diff[denominator == 0] = 0.0
    return diff

y_pred = np.array([1, 5.999])
y_true = np.array([3, 6])
smape(y_pred, y_true )

# smape grad
# input y_pred as np.array
#       y_true as np.array
def smape_grad(y_pred, y_true):
    f_prime = np.array([1.0 if check else -1.0 for check in y_pred >= y_true])
    g_prime = np.array([1.0 if check else -1.0 for check in y_pred >= 0.0])
    f_x = np.abs(y_true - y_pred)
    g_x = (np.abs(y_true)+np.abs(y_pred)) 
    return 2 * (f_prime * g_x - f_x * g_prime) / np.square(g_x)

y_pred = np.array([5.999])
y_true = np.array([6])
smape_grad(y_pred, y_true)

nd.Gradient(partial(smape, y_true=y_true))(y_pred[i])

from scipy.optimize import approx_fprime
def smape_hess(y_pred, y_true):
    eps = np.sqrt(np.finfo(float).eps)
    res = np.array([approx_fprime(np.array([y_pred[i]]), \
                        partial(smape_grad,y_true=np.array([y_true[i]])), eps) \
                    for i in range(y_pred.size)])
    return np.concatenate(res)
    

y_pred = np.array([1, 1.9])
y_true = np.array([3, 2])
smape_grad(y_pred, y_true)
smape_hess(y_pred, y_true)
 
from scipy.optimize import check_grad
from functools import partial
import numdifftools as nd

for i in range(y_pred.size):
    print nd.Gradient(partial(smape, y_true=y_true[i]))(i)
    print nd.Hessian(partial(smape, y_true=y_true[i]))(y_pred[i])



p_smape = partial(smape, y_true=y_true)
p_smape_grad = partial(smape_grad,y_true=y_true)
p_smape_hess = partial(smape_hess,y_true=y_true)
check_grad(p_smape, p_smape_grad, [1])
check_grad(p_smape, p_smape_grad, [15])
check_grad(p_smape_grad, p_smape_hess, [15])


y_true = 3 * np.ones(10)
y_pred = np.array(range(10))


print smape_grad(y_pred, y_true=y_true)
print smape_hess(y_pred, y_true=y_true)

def mae(y):
    return y[1]

def mae_grad(y):
    return 1.0
    
check_grad(mae,mae_grad,[3,1])


def func(x):
     return x[0]**2 - 0.5 * x[1]**3
def grad(x):
     return [2 * x[0], -1.5 * x[1]**2]
check_grad(func, grad,[1.5, -1.5])
