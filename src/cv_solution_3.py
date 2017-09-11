#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 20:58:57 2017

@author: eyulush
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import lightgbm as lgb
import copy

from common import parse_page
from common import get_language

from common import TrainingCtrl, smape

train = pd.read_csv("../input/train_2.csv.zip",compression='zip')
train = train.fillna(0.)

# debug part
train_samples = train.sample(200)

train_samples=pd.merge(train_samples.apply(lambda x: parse_page(x['Page'], ret_series=True), axis=1), train_samples,
                   how='right', left_index=True, right_index=True)
train_samples.drop('article',inplace=True,axis=1)