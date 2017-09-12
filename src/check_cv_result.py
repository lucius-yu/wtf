#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:28:33 2017

@author: eyulush
"""
import numpy as np
import pandas as pd

# 1st validation result check
# 16 days
df_result_1=pd.read_csv('../output/cv_solution_3_with_round_2016-09-14_2016-11-14',index_col='date')
df_result_1['s2_score'] = pd.read_csv('../output/cv_solution_2_with_round_2016-09-14_2016-11-14', header=None ).values
df_result_1.plot(figsize=(10,8))

gain=(df_result_1['s2_score'] - df_result_1['score']).cumsum()
print(np.argmax(gain.values))

# 2nd validation result check
# 17 days
df_result_2=pd.read_csv('../output/cv_solution_3_with_round_2016-10-12_2016-12-12',index_col='date')
df_result_2['s2_score'] = pd.read_csv('../output/cv_solution_2_with_round_2016-10-12_2016-12-12', header=None ).values
df_result_2.plot(figsize=(10,8))
gain=(df_result_2['s2_score'] - df_result_2['score']).cumsum()
print(np.argmax(gain.values))


# 3rd validation result check
# 17 days
df_result_3=pd.read_csv('../output/cv_solution_3_with_round_2017-03-15_2017-05-15',index_col='date')
df_result_3['s2_score'] = pd.read_csv('../output/cv_solution_2_with_round_2017-03-15_2017-05-15', header=None ).values
df_result_3.plot(figsize=(10,8))
gain=(df_result_3['s2_score'] - df_result_3['score']).cumsum()
print(np.argmax(gain.values))


# 4th validation result check
df_result_4=pd.read_csv('../output/cv_solution_3_with_round_2017-05-10_2017-07-10',index_col='date')
df_result_4['s2_score'] = pd.read_csv('../output/cv_solution_2_with_round_2017-05-10_2017-07-10', header=None ).values
df_result_4.plot(figsize=(10,8))
gain=(df_result_4['s2_score'] - df_result_4['score']).cumsum()
print(np.argmax(gain.values))