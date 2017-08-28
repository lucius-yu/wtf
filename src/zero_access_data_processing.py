# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:23:01 2017

@author: eyulush
"""

from functools import partial
import pandas as pd
import numpy as np
import re
import common
from sklearn.metrics import confusion_matrix

# control variable
train_processing = False
valid_processing = False
test_processing = True
submit_processing = True

langs = ['zh', 'fr', 'en', 'na', 'ru', 'de', 'ja', 'es']

# support function
def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res.group(0)[0:2]
    return 'na'

'''
x should be pd.series
'''
def get_zero_access_rate(x):
    return float((x==0).sum())/x.notnull().sum() if x.notnull().sum() > 0 else 0.0

def get_dow(datetime_str):
    return pd.to_datetime(datetime_str).weekday()

def get_dom(datetime_str):
    return pd.to_datetime(datetime_str).day

def get_doy(datetime_str):
    return pd.to_datetime(datetime_str).dayofyear

def get_moy(datetime_str):
    return pd.to_datetime(datetime_str).month

# x is a date
def get_nonworking_day(date, lang):
    #official non working days by country (manual search with google)
    #I made a lot of shortcuts considering that only Us and Uk used english idiom, 
    #only Spain for spanich, only France for french, etc
    train_us=['2015-07-04','2015-11-26','2015-12-25','2016-07-04','2016-11-24','2016-12-26']
    test_us=[]
    train_uk=['2015-12-25','2015-12-28','2016-01-01','2016-03-28','2016-05-02','2016-05-30','2016-12-26','2016-12-27']
    test_uk=['2017-01-01']
    train_de=['2015-10-03', '2015-12-25', '2015-12-26','2016-01-01', '2016-03-25', '2016-03-26', '2016-03-27', '2016-01-01', '2016-05-05', '2016-05-15', '2016-05-16', '2016-10-03', '2016-12-25', '2016-12-26']
    test_de=['2017-01-01']
    train_fr=['2015-07-14', '2015-08-15', '2015-11-01', '2015-11-11', '2015-12-25','2016-01-01','2016-03-28', '2016-05-01', '2016-05-05', '2016-05-08', '2016-05-16', '2016-07-14', '2016-08-15', '2016-11-01','2016-11-11', '2016-12-25']
    test_fr=['2017-01-01']
    train_ru=['2015-11-04','2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04', '2016-01-05', '2016-01-06', '2016-01-07', '2016-02-23', '2016-03-08', '2016-05-01', '2016-05-09', '2016-06-12', '2016-11-04']
    test_ru=['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07', '2017-02-23']
    train_es=['2015-08-15', '2015-10-12', '2015-11-01', '2015-12-06', '2015-12-08', '2015-12-25','2016-01-01', '2016-01-06', '2016-03-25', '2016-05-01', '2016-08-15', '2016-10-12', '2016-11-01', '2016-12-06', '2016-12-08', '2016-12-25']
    test_es=['2017-01-01', '2017-01-06']
    train_ja=['2015-07-20','2015-09-21', '2015-10-12', '2015-11-03', '2015-11-23', '2015-12-23','2016-01-01', '2016-01-11', '2016-02-11', '2016-03-20', '2016-04-29', '2016-05-03', '2016-05-04', '2016-05-05', '2016-07-18', '2016-08-11', '2016-09-22', '2016-10-10', '2016-11-03', '2016-11-23', '2016-12-23']
    test_ja=['2017-01-01', '2017-01-09', '2017-02-11']
    train_zh=['2015-09-27', '2015-10-01', '2015-10-02','2015-10-03','2015-10-04','2015-10-05','2015-10-06','2015-10-07','2016-01-01', '2016-01-02', '2016-01-03', '2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12', '2016-04-04', '2016-05-01', '2016-05-02', '2016-06-09', '2016-06-10', '2016-09-15', '2016-09-16', '2016-10-03', '2016-10-04','2016-10-05','2016-10-06','2016-10-07']
    test_zh=['2017-01-02', '2017-02-27', '2017-02-28', '2017-03-01']
    #in China some saturday and sundays are worked, so the following date are not not working day
    train_o_zh=['2015-10-10','2016-02-06', '2016-02-14', '2016-06-12', '2016-09-18', '2016-10-08', '2016-10-09']
    test_o_zh=['2017-01-22', '2017-02-04']
    holiday_dict = {'en': train_us + train_uk + test_us + test_uk,
                    'de': train_de + test_de,
                    'fr': train_fr + test_fr,
                    'ru': train_ru + test_ru,
                    'es': train_es + test_es,
                    'ja': train_ja + test_ja,
                    'zh': train_zh + test_zh,
                    'na': [],
                    'o_zh': train_o_zh + test_o_zh}
    if lang=='zh' and (date in holiday_dict['o_zh']):
        return 0
    if get_dow(date) >= 5 or (date in holiday_dict[lang]):
        return 1
    return 0

# wrapper    
def is_nonworking_day(x, lang=None):
    if lang==None:
        lang = x.lang
    return get_nonworking_day(x.date, lang)

# flatten the train data for binary classification
def expand_page_date_label(df_data):
    df_result = pd.DataFrame()
    
    for index, row in df_data.iterrows():
        df_result=df_result.append(pd.DataFrame([{'Page': row.Page, 'date': date, \
                                                  'lang': row.lang,'label' : row[date]==0} \
                                                   for date in row.keys()[2:]]))    
    return df_result


# load train
train = pd.read_csv('../input/train_1.csv.zip',compression='zip')

# add column for language
train.insert(loc=1, column='lang',value=train.Page.map(get_language))

### first train-test-split.
n_prior = 430
n_valid = 60
n_test = 60

df_prior = train.iloc[:,0:n_prior+2]
df_valid = train.iloc[:, [0,1]+range(n_prior+2, n_prior+n_valid+2)]
df_test = train.iloc[:, [0,1]+range(n_prior+n_valid+2,n_prior+n_valid+n_test+2)]

### do statistics

### page related properities
page_stats_keys = ['Page', 'lang', 'zero_access_rate', 'mean_access', 'median_access', 'std_access']

# identfy zero access day / total avail days, the number higher --> high probability of 0 access on that day
zero_access_rate = train.iloc[:,2:].apply(get_zero_access_rate, axis=1).rename('zero_access_rate')
mean_access = train.iloc[:,2:].apply(lambda x: x.mean(),axis=1).rename('mean_access')
median_access = train.iloc[:,2:].apply(lambda x: x.median(),axis=1).rename('median_access')
std_access = train.iloc[:,2:].apply(lambda x: x.std(),axis=1).rename('std_access')

page_stats = pd.concat([train[['Page','lang']],  zero_access_rate, mean_access, median_access, std_access],axis=1)

common.save_df(page_stats,'../data/','page_stats')

### date related properities, day of week, day of month, month of year, day of year
date_keys = ['date', 'dow', 'dom', 'moy', 'doy']
date_prop = pd.DataFrame({'date':train.keys()[2:], 
                          'dow': train.keys()[2:].map(get_dow),
                          'dom': train.keys()[2:].map(get_dom),
                          'moy': train.keys()[2:].map(get_moy),
                          'doy': train.keys()[2:].map(get_doy)})
    
common.save_df(date_prop,'../data/','date_prop')

### lang and langXdate properties
date_lang_keys = ['date','lang','non_working_day']
date_lang_prop = pd.DataFrame(columns=['date','lang'])

for lang in langs:
    dlp = pd.DataFrame(date_prop['date'])
    dlp['lang'] = pd.Series([lang]*dlp.shape[0])
    date_lang_prop = date_lang_prop.append(dlp,ignore_index=True)    

date_lang_prop['non_working_day'] = date_lang_prop.apply(is_nonworking_day,axis=1)

common.save_df(date_lang_prop,'../data/','date_lang_prop')

### cross stats
# page weekday zero access rate, page weekday mean access rate
train_list = common.split_dataframe(train,n=8)

'''
outout:  page, weekday, weekday_mean_access_rate, weekday_zero_access_rate
 
'''
def get_one_weekday_stats(df_row):
    dow_grouped = df_row[2:].groupby(get_dow)
    return pd.DataFrame([{'Page':df_row[0],'weekday':name,\
                          'weekday_mean_access':group.mean(),\
                          'weekday_zero_access_rate': get_zero_access_rate(group)} \
                        for name, group in dow_grouped])

def get_weekday_stats(df_data):
    return pd.concat([get_one_weekday_stats(row) \
                      for index, row in df_data.iterrows()],ignore_index=True)

page_weekday_stats = common.papply(train_list, get_weekday_stats)

common.save_df(page_weekday_stats,'../data/','page_weekday_stats')


'''
output: page, non_working_day, nw_day_mean_access, nw_day_zero_access_rate, 
'''
def get_one_non_working_day_stats(df_row):
    nw_grouped = df_row[2:].groupby(partial(get_nonworking_day, lang=df_row[1]))
    return pd.DataFrame([{'Page' : df_row[0],'non_working_day' : name,
                          'nw_day_mean_access': group.mean(),
                          'nw_day_zero_access_rate': get_zero_access_rate(group)} \
                        for name, group in nw_grouped])

def get_non_working_day_stats(df_data):
    return pd.concat([get_one_non_working_day_stats(row) \
                      for index, row in df_data.iterrows()],ignore_index=True)
    
page_non_working_day_stats = common.papply(train_list, get_non_working_day_stats)

common.save_df(page_non_working_day_stats,'../data/','page_non_working_day_stats')

# setup training data
# valid set start as df_valid.keys()[2], 2016-09-03
if train_processing:
    train_idx_1 = np.where(train.keys()=='2015-09-03')[0][0]
    train_idx_2 = np.where(train.keys()=='2016-08-03')[0][0]
    df_train = train.iloc[:, [0,1] + range(train_idx_1, train_idx_1+30) \
                      + range(train_idx_2, train_idx_2+30)]

    df_train = df_train.dropna()
    df_train_list = common.split_dataframe(df_train,n=8)
    df_train = common.papply(df_train_list, expand_page_date_label)

    # merge the properties
    df_train = df_train.merge(page_stats,how='left',on=['Page','lang'])
    df_train = df_train.merge(date_prop, how='left',on='date')
    df_train = df_train.merge(date_lang_prop,how='left',on=['date','lang'])

    df_train = df_train.merge(page_weekday_stats, how='left',
                              left_on=['Page','dow'], right_on=['Page', 'weekday'])
    df_train = df_train.merge(page_non_working_day_stats, how='left',
                              on=['Page','non_working_day'])

    df_train['lang'] = df_train['lang'].astype('category')
    df_train['non_working_day'] = df_train['non_working_day'].astype('category')
    df_train['weekday'] = df_train['weekday'].astype('category')

    common.save_df(df_train,'../data/','df_train')

if valid_processing:
    df_valid = df_valid.dropna()
    df_valid_list = common.split_dataframe(df_valid,n=8)
    df_valid = common.papply(df_valid_list, expand_page_date_label)

    df_valid = df_valid.merge(page_stats,how='left',on=['Page','lang'])
    df_valid = df_valid.merge(date_prop, how='left',on='date')
    df_valid = df_valid.merge(date_lang_prop,how='left',on=['date','lang'])

    df_valid = df_valid.merge(page_weekday_stats, how='left',
                              left_on=['Page','dow'], right_on=['Page', 'weekday'])
    df_valid = df_valid.merge(page_non_working_day_stats, how='left',
                              on=['Page','non_working_day'])

    df_valid['lang'] = df_valid['lang'].astype('category')
    df_valid['non_working_day'] = df_valid['non_working_day'].astype('category')
    df_valid['weekday'] = df_valid['weekday'].astype('category')

    common.save_df(df_valid,'../data/','df_valid')

if test_processing:
    df_test = df_test.dropna()
    df_test_list = common.split_dataframe(df_test,n=8)
    df_test = common.papply(df_test_list, expand_page_date_label)

    df_test = df_test.merge(page_stats,how='left',on=['Page','lang'])
    df_test = df_test.merge(date_prop, how='left',on='date')
    df_test = df_test.merge(date_lang_prop,how='left',on=['date','lang'])

    df_test = df_test.merge(page_weekday_stats, how='left',
                              left_on=['Page','dow'], right_on=['Page', 'weekday'])
    df_test = df_test.merge(page_non_working_day_stats, how='left',
                              on=['Page','non_working_day'])

    df_test['lang'] = df_test['lang'].astype('category')
    df_test['non_working_day'] = df_test['non_working_day'].astype('category')
    df_test['weekday'] = df_test['weekday'].astype('category')

    common.save_df(df_test,'../data/','df_test')
    
if submit_processing:
    # setup feature order is important
    df_sub = pd.read_csv("../input/key_1.csv.zip",compression='zip')
    df_sub['date'] = df_sub.Page.apply(lambda a: a[-10:])
    df_sub['Page'] = df_sub.Page.apply(lambda a: a[:-11])
    df_sub['lang'] = df_sub.Page.map(get_language)
    df_sub = df_sub.merge(df_sub.date.agg([get_dow, get_dom, get_doy, get_moy]).rename(\
                          columns={'get_dow': 'dow','get_dom': 'dom','get_doy': 'doy','get_moy': 'moy'}),\
                          how='left', left_index=True, right_index=True)

    df_sub['non_working_day'] = df_sub.apply(lambda row: get_nonworking_day(row.date, row.lang),axis=1)

    df_sub = df_sub.merge(page_stats,how='left',on=['Page','lang'])
    df_sub = df_sub.merge(page_weekday_stats, how='left',
                              left_on=['Page','dow'], right_on=['Page', 'weekday'])
    df_sub = df_sub.merge(page_non_working_day_stats, how='left',
                              on=['Page','non_working_day'])

    df_sub['lang'] = df_sub['lang'].astype('category')
    df_sub['non_working_day'] = df_sub['non_working_day'].astype('category')
    df_sub['weekday'] = df_sub['weekday'].astype('category')

    common.save_df(df_sub,'../data/','df_sub')