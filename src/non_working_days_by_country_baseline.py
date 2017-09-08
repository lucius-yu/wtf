#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:36:50 2017

Baseline for WTF, 
  working day and non-working day median solution

Training data
  Each page is from 2015-07-01 to 2017-08-31
Test data
  Each page is from 2017-09-13 to 2017-11-13. 2 Months (62 days) prediction

@author: eyulush
"""

import pandas as pd
import numpy as np
import re
import gc; gc.enable()

from sklearn.feature_extraction import text
from sklearn import naive_bayes

from common import get_holidays

pd.options.display.max_rows = 999
pd.options.display.max_columns = 100

'''
### Working day and Non working day solution
'''

'''
1. train data processing
'''
train = pd.read_csv("../input/train_2.csv.zip",compression='zip')
train = train.fillna(0.)

train['origine']=train['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])

#let's make a train, target, and test to predict language on ts and er pages
orig_train=train.loc[~train.origine.isin(['ts', 'er']), 'Page']
orig_target=train.loc[~train.origine.isin(['ts', 'er']), 'origine']
orig_test=train.loc[train.origine.isin(['ts', 'er']), 'Page']
#keep only interesting chars
orig_train2=orig_train.apply(lambda x:x.split(".wikipedia")[0][:-3]).apply(lambda x:re.sub("[a-zA-Z0-9():\-_ \'\.\/]", "", x))
orig_test2=orig_test.apply(lambda x:x.split(".wikipedia")[0][:-3]).apply(lambda x:re.sub("[a-zA-Z0-9():\-_ \'\.\/]", "", x))
#run TFIDF on those specific chars
tfidf=text.TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, 
                     lowercase=True, preprocessor=None, tokenizer=None, 
                     analyzer='char', #stop_words=[chr(x) for x in range(97,123)]+[chr(x) for x in range(65,91)]+['_','.',':'], 
                     token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=True, norm='l2', 
                     use_idf=True, smooth_idf=True, sublinear_tf=False)
orig_train2=tfidf.fit_transform(orig_train2)
#apply a simple naive bayes on the text features
model=naive_bayes.BernoulliNB()
model.fit(orig_train2, orig_target)
result=model.predict(tfidf.transform(orig_test2))
result=pd.DataFrame(result, index=orig_test)
result.columns=['origine']

del train['origine']

''' let's flatten the train as did clustifier and initialize a "ferie" columns instead of a weekend column '''
days_to_use = 49 # how many latest days is used for solution
train = pd.melt(train[list(train.columns[-days_to_use:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
train['date'] = train['date'].astype('datetime64[ns]')
train['ferie'] = ((train.date.dt.dayofweek) >=5).astype(float)
train['origine']=train['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])

#let's join with result to replace 'ts' and 'er'
join=train.loc[train.origine.isin(["ts","er"]), ['Page']]
join['origine']=0 #init
join.index=join["Page"]
join.origine=result
train.loc[train.origine.isin(["ts","er"]), ['origine']]=join.origine.values #replace

''' official holidays '''
holiday_us, holiday_uk, holiday_de, holiday_fr, holiday_ru, holiday_es, holiday_ja, holiday_zh, holiday_o_zh = \
    get_holidays()
    
''' replace ferie '''
train.loc[(train.origine=='en')&(train.date.isin(holiday_us+holiday_uk)), 'ferie']=1
train.loc[(train.origine=='de')&(train.date.isin(holiday_de)), 'ferie']=1
train.loc[(train.origine=='fr')&(train.date.isin(holiday_fr)), 'ferie']=1
train.loc[(train.origine=='ru')&(train.date.isin(holiday_ru)), 'ferie']=1
train.loc[(train.origine=='es')&(train.date.isin(holiday_es)), 'ferie']=1
train.loc[(train.origine=='ja')&(train.date.isin(holiday_ja)), 'ferie']=1
train.loc[(train.origine=='zh')&(train.date.isin(holiday_zh)), 'ferie']=1
train.loc[(train.origine=='zh')&(train.date.isin(holiday_o_zh)), 'ferie']=0

'''
2. test data processing
'''
test = pd.read_csv("../input/key_2.csv.zip", compression='zip')

test['date'] = test.Page.apply(lambda a: a[-10:]) # get the date
test['Page'] = test.Page.apply(lambda a: a[:-11]) # get the page
test['date'] = test['date'].astype('datetime64[ns]')
test['ferie'] = ((test.date.dt.dayofweek) >=5).astype(float)
test['origine']=test['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])

#joint with result
join=test.loc[test.origine.isin(["ts","er"]), ['Page']]
join['origine']=0
join.index=join["Page"]
join.origine=result
test.loc[test.origine.isin(["ts","er"]), ['origine']]=join.origine.values

test.loc[(test.origine=='en')&(test.date.isin(holiday_us+holiday_uk)), 'ferie']=1
test.loc[(test.origine=='de')&(test.date.isin(holiday_de)), 'ferie']=1
test.loc[(test.origine=='fr')&(test.date.isin(holiday_fr)), 'ferie']=1
test.loc[(test.origine=='ru')&(test.date.isin(holiday_ru)), 'ferie']=1
test.loc[(test.origine=='es')&(test.date.isin(holiday_es)), 'ferie']=1
test.loc[(test.origine=='ja')&(test.date.isin(holiday_ja)), 'ferie']=1
test.loc[(test.origine=='zh')&(test.date.isin(holiday_zh)), 'ferie']=1
test.loc[(test.origine=='zh')&(test.date.isin(holiday_o_zh)), 'ferie']=0

'''
3. find the solution, i.e. median on latest 49 days
'''
train_page_per_dow = train.groupby(['Page','ferie']).median().reset_index()
test = test.merge(train_page_per_dow, how='left')

test.loc[test.Visits.isnull(), 'Visits'] = 0
test['Visits']=(test['Visits']*10+np.random.randint(0,2,len(test['Visits']))).astype('int')/10
# test['Visits']=((test['Visits']*10).astype('int')/10 + test1['Visits'])/2
test[['Id','Visits']].to_csv('../submit/res.csv', index=False)