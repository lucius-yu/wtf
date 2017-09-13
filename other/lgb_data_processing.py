import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re


train = pd.read_csv("../input/train_1.csv.zip",compression='zip')

# common prepocessing
def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res.group(0)[0:2]
    return 'na'

# add column for language
train.insert(loc=1, column='lang',value=train.Page.map(get_language))

# train test split
n_valid = 60

df_train = train.iloc[:,:-n_valid]
df_valid = train.iloc[:,[0,1]+list(range(train.shape[1]-n_valid, train.shape[1]))]

'''
# data processing
DayWindows = [7, 11, 18, 30, 48, 78, 126, 203, 329]
WeekDayWindows = [5, 7, 9]
WorkingDayWindows = [5, 15, 30, 60, 120]
NonWorkingDayWindows = [4, 8, 16, 32]
1. mean visits
2. median visits
3. std visits
4. zero visits
'''
DayWindows = [7, 11, 18, 30, 48, 78, 126, 203, 329]
WeekDayWindows = [5, 7, 9]

for idx, row in df_train.iterrows():
    # jump the start page and lang
    series = row[2:]
    # remove initial 0 access
    series = series[series.notnull().cumsum()>0]
    
    series.index=pd.to_datetime(series.index)
    