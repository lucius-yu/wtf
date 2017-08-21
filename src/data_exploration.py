# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 12:57:29 2017

@author: eyulush
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# load train
train = pd.read_csv('../input/train_1.csv.zip',compression='zip').fillna(0)
train.head()

# downcast to integer
for col in train.columns[1:]:
    train[col]=train[col].astype(np.int16)

# traffic related to language ?
def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res.group(0)[0:2]
    return 'na'

# add column for language
train['lang'] = train.Page.map(get_language)

# better edition just cut and get language 
train['origine']=train['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])


'''
There are 7 languages plus the media pages. The languages used here are: 
    English, Japanese, German, French, Chinese, Russian, and Spanish.
  
'''
train.groupby('lang').size()
train.groupby('origine').size()

# using fft to analyse pages by page lanuage
lang_sets = {}
lang_sets['en'] = train[train.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = train[train.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = train[train.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = train[train.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = train[train.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = train[train.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = train[train.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = train[train.lang=='es'].iloc[:,0:-1]

sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]
    
# first, just plot the raw information by language
days = [r for r in range(sums['en'].shape[0])]

fig = plt.figure(1,figsize=[10,10])
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
labels={'en':'English','ja':'Japanese','de':'German',
        'na':'Media','fr':'French','zh':'Chinese',
        'ru':'Russian','es':'Spanish'}

for key in sums:
    plt.plot(days,sums[key],label = labels[key] )
    
plt.legend()
plt.show()


# define figure size
fs = (12,8)
# how about just rotate the dataframe, groupby languange, do the sum then plot 
train_tt = (train.drop('Page',axis=1).groupby('lang').sum()).transpose()
train_tt.plot(figsize=fs)

'''
Comments: as summarized by language, 
1. there is clear point, 
    we have short term period e.g. one week.
    we have long term period e.g. month, or a year
2. as time going, generally the access will be more    
3. note for special days or events. e.g. chinese spring festival etc.
    
'''

# first, let's check short term period
# clearly, for europe, the on saturday, the wiki access has lowest number.
train_tt[['de','es','fr']].iloc[:14].sum(axis=1).plot(figsize=(12,8))
train_tt[['de','es','fr']].iloc[10*7:12*7].sum(axis=1).plot(figsize=(12,8))
train_tt[['de','es','fr']].iloc[20*7:22*7].sum(axis=1).plot(figsize=(12,8))

# for asia, it is oppsite, they go to wiki on weekend, timezon difference?
train_tt[['zh','ja']].iloc[:14].sum(axis=1).plot(figsize=(12,8))
train_tt[['zh','ja']].iloc[10*7:12*7].sum(axis=1).plot(figsize=(12,8))
train_tt[['zh','ja']].iloc[30*7:32*7].sum(axis=1).plot(figsize=(12,8))

# for en and total, there is also shown period on week
train_tt[['en','na']].iloc[:14].sum(axis=1).plot(figsize=(12,8))
train_tt[['en','na']].iloc[10*7:12*7].sum(axis=1).plot(figsize=(12,8))
train_tt[['en','na']].iloc[30*7:32*7].sum(axis=1).plot(figsize=(12,8))

train_tt[['de','es','fr']].iloc[:60].sum(axis=1).plot(figsize=fs)

''' 
also, some event will happen yearly, e.g. chinese spring festival will impact on zh page access.
so choose proper period for prediction is important.
'''

# second, let's sum on weeks and show 8 weeks 
'''
    it shows es has different pattern. On week 27 + 25 = 52, there is a big drop
'''
train_tt.groupby(train_tt.reset_index(drop=False).index // 7).sum(axis=1).plot(figsize=fs)

# second, let's sum on weeks and show 8 weeks 
'''
    other information for weekday 
'''
train_tt.groupby(train_tt.index.to_datetime().weekday).sum(axis=1)



''' 
another exploration
'''
train = pd.read_csv('../input/train_1.csv.zip',index_col='Page', compression='zip').fillna(0)
train_tt = train.transpose()
train_tt.index=train_tt.index.to_datetime()

train_tt.iloc[:,2000].groupby(train_tt.index.weekday).sum()
train_tt.iloc[-28:,2000].plot(figsize=fs)

import matplotlib.pyplot as plt
from scipy.fftpack import dct,idct

test_page = 2004
period = 14
window = 7
x = train_tt.iloc[-period:,test_page].tolist()
dct_x=dct(x,1, )
dct_x=dct_x[:window].tolist()+[0]*(period-window+period)

idct_x = idct(dct_x)
plt.plot(idct_x)
plt.plot(x)

