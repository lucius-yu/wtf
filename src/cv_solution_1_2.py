'''
Conclusion:
    1. use the round to int will get better result.
    2. solution 2 better than solution 1
'''


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os
import lightgbm as lgb

from common import parse_page
from common import get_language
from common import TrainingCtrl, smape


def generate_train_data(df_data, start_date, end_date):
    cols = ['Page'] + list(map(str,pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d')))
    df_train = df_data.loc[:, cols]
    return df_train

def generate_test_data(df_data, start_date, end_date):
    test_dates = list(map(str,pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d')))
    df_test = df_data.Page.apply(lambda x: pd.Series([str(x) + '_' + test_date for test_date in test_dates]))
    df_test = pd.melt(df_test, value_name='Page', var_name='Id')
    return df_test

def generate_real_data(df_data, start_date, end_date):
    cols = ['Page'] + list(map(str,pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d')))
    df_real = pd.melt(df_data.loc[:,cols], id_vars='Page', var_name='date', value_name='Real_Visits')
    return df_real

def get_smape_scores(df_real, df_pred, round_flag=False):
    f_keys = ['Page', 'date']
    df_valid = pd.merge(left=df_real[f_keys + ['Real_Visits']], right=df_pred[f_keys + ['Visits']], on=f_keys, how='left')
    if round_flag:
        return df_valid.groupby('date').apply(lambda x: smape(x['Real_Visits'], round(x['Visits'])))
    return df_valid.groupby('date').apply(lambda x: smape(x['Real_Visits'], x['Visits']))
    
# we need to make solution
# solution 1 working day, non-working day solution
def solution_1(df_train, df_test):
    from sklearn.feature_extraction import text
    from sklearn import naive_bayes
    from common import get_holidays
    
    train = df_train.copy()
    test = df_test.copy()

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

    train_page_per_dow = train.groupby(['Page','ferie']).median().reset_index()
    test = test.merge(train_page_per_dow, how='left')

    test.loc[test.Visits.isnull(), 'Visits'] = 0
    test['Ori_Visits'] = test['Visits']
    test['Visits']=(test['Visits']*10+np.random.randint(0,2,len(test['Visits']))).astype('int')/10
    test['date'] = test['date'].astype(str)
    return test

# solution_2 is based on solution_1
def solution_2(df_train, df_test, s1_result=None):
    train = df_train.copy()
    test = df_test.copy()
    
    if s1_result is None:
        print('s1_result is none')
        s1_result = solution_1(train, test)
        
    # solution 2 part
    Windows = [11, 18, 30, 48, 78, 126, 203, 329]

    n = train.shape[1] - 1 #  
    Visits = np.zeros(train.shape[0])
    for i, row in train.iterrows():
        M = []
        start = row[1:].nonzero()[0]
        if len(start) == 0:
            continue
        if n - start[0] < Windows[0]:
            Visits[i] = row.iloc[start[0]+1:].median()
            continue
        for W in Windows:
            if W > n-start[0]:
                break
            M.append(row.iloc[-W:].median())
        Visits[i] = np.median(M)

    # update the Visits
    Visits[np.where(Visits < 1)] = 0.
    train['Visits'] = Visits

    test1 = test.copy()
    test1['Page'] = test1.Page.apply(lambda x: x[:-11])
    test1 = test1.merge(train[['Page','Visits']], on='Page', how='left')
    
    s2_result = s1_result[['Page', 'date', 'Id']].copy()
    s2_result['Visits']=((s1_result['Ori_Visits']*10).astype('int')/10 + test1['Visits'])/2
    return s2_result
    
    
# cross-validation date
# real test start date is 2017-09-13, Wensday. 
# candidate:  2016-09-14 : 2016-11-14, 2017-03-15:2017-05-15
cross_validation = False
submission = True

# load date
train = pd.read_csv("../input/new_train_2.csv.zip",compression='zip')
train = train.fillna(0.)


if cross_validation:
    valid_dates = [('2015-07-01', '2016-09-13', '2016-09-14', '2016-11-14'), 
                   ('2015-07-01', '2017-03-14', '2017-03-15', '2017-05-15'),
                   ('2015-07-01', '2016-10-11', '2016-10-12', '2016-12-12'), 
                   ('2015-07-01', '2017-05-09', '2017-05-10', '2017-07-10')]


    cv_scores = pd.read_csv('../output/cv_scores.csv') if os.path.isfile('../output/cv_scores.csv') else pd.DataFrame()

    for train_start_date, train_end_date, valid_start_date, valid_end_date in valid_dates:
        # solution 1 validation
        # solution 1 will have 'Visits' and 'Visits_1' 
        # Visits_1 is mixed with randint
        df_train = generate_train_data(train, train_start_date, train_end_date)
        df_test = generate_test_data(train, start_date=valid_start_date, end_date=valid_end_date)
        df_real = generate_real_data(train, start_date=valid_start_date, end_date=valid_end_date)

        s1_result = solution_1(df_train, df_test)
        s1_scores = get_smape_scores(df_real, s1_result, round_flag=False)
        print("solutions_1 without round scores %f"%s1_scores.mean())
        s1_scores.to_csv('../output/cv_solution_1_without_round_' + valid_start_date + '_' + valid_end_date, index=False)
        cv_scores = cv_scores.append(pd.DataFrame({'solution': ['solution_1'], 'round': [False],
                                'valid_start_date': [valid_start_date], 'score': [s1_scores.mean()]}))
    
        s1_scores = get_smape_scores(df_real, s1_result, round_flag=True)
        print("solutions_1 with round scores %f"%s1_scores.mean())
        s1_scores.to_csv('../output/cv_solution_1_with_round_' + valid_start_date + '_' + valid_end_date, index=False)
        cv_scores = cv_scores.append(pd.DataFrame({'solution':['solution_1'], 'round': [True],
                                                   'valid_start_date': [valid_start_date], 'score': [s1_scores.mean()]}))
   

        # solution 2 validation, use copy to keep train not modified
        s2_result = solution_2(df_train, df_test, s1_result)
        s2_scores = get_smape_scores(df_real, s2_result, round_flag=False)
        print("solutions_2 without round scores %f"%s2_scores.mean())
        s2_scores.to_csv('../output/cv_solution_2_without_round_' + valid_start_date + '_' + valid_end_date, index=False)
        cv_scores = cv_scores.append(pd.DataFrame({'solution': ['solution_2'], 'round': [False],
                                                   'valid_start_date': [valid_start_date], 'score': [s2_scores.mean()]}))
   
    
        s2_scores = get_smape_scores(df_real, s2_result, round_flag=True)
        print("solutions_2 with round scores %f"%s2_scores.mean())
        s2_scores.to_csv('../output/cv_solution_2_with_round_' + valid_start_date + '_' + valid_end_date, index=False)
        cv_scores = cv_scores.append(pd.DataFrame({'solution':['solution_2'], 'round': [True],
                                                   'valid_start_date': [valid_start_date], 'score': [s2_scores.mean()]}))

    cv_scores.to_csv('../output/cv_scores.csv',index=False)

if submission:
    # real solution
    real_test = pd.read_csv("../input/key_2.csv.zip",compression='zip')
    '''
    # Be careful
    # we might not have enough data from training
    START_DATE='20170901'
    END_DATE='20170910'

    extra_train = pd.read_csv('../input/extra_train_'+START_DATE+'_'+END_DATE+'.csv.zip')
    print(extra_train.iloc[:,-1].isnull().sum())
    print(extra_train.iloc[:,-2].isnull().sum())
    print(extra_train.iloc[:,-3].isnull().sum())
    
    real_train = train.merge(extra_train,how='left', on='Page')
    real_train.fillna(0.0)
    '''
    
    s1_result = solution_1(train, real_test)
    s2_result = solution_2(train, real_test, s1_result)
    
    s1_result['Visits'] = round(s1_result['Visits'])
    s2_result['Visits'] = round(s2_result['Visits'])
    
    s1_result.to_csv('../submit/solution_1_raw.csv',index=False)
    s2_result.to_csv('../submit/solution_2_raw.csv',index=False)