'''
weekend flag kernel

https://www.kaggle.com/chechir/weekend-flag-median-with-wiggle

score : 46.1

'''

import pandas as pd

def get_raw_data():
    train = pd.read_csv("../input/train_1.csv.zip",compression='zip')
    test = pd.read_csv("../input/key_1.csv.zip",compression='zip')
    return train, test

# just mark if a day is weekend, 
# periods is used to cut the data for not doing stats
def transform_data(train, test, periods=-49):
    train_flattened = pd.melt(train[list(train.columns[periods:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
    train_flattened = get_features(train_flattened)
    test['date'] = test.Page.apply(lambda a: a[-10:])
    test['Page'] = test.Page.apply(lambda a: a[:-11])
    test = get_features(test)
    return train_flattened, test

def get_features(df):
    df['date'] = df['date'].astype('datetime64[ns]')
    df['every2_weekdays'] = df.date.dt.dayofweek
    df['weekend'] = (df.date.dt.dayofweek // 5).astype(float)
    #df['shortweek'] = ((df.date.dt.dayofweek) // 4 == 1).astype(float)
    return df

# get median of weekend
def predict_using_median_weekend(train, test):
    df = train.copy()
    df = df.drop(['every2_weekdays'], axis=1)
    agg_train_weekend = df.groupby(['Page', 'weekend']).median().reset_index()
    test_df = test.merge(agg_train_weekend, how='left')
    result = test_df['Visits'].values
    return result

def predict_using_median_weirddays(train, test):
    df = train.copy()
    df = df.drop(['weekend'], axis=1)
    agg_train_weekend = df.groupby(['Page', 'every2_weekdays']).median().reset_index()
    test_df = test.merge(agg_train_weekend, how='left')
    result = test_df['Visits'].values
    return result

def wiggle_preds(df):
    second_term_ixs = df['date'] < '2017-02-01'
    adjusted = df['Visits'].values + df['Visits'].values*0.02
    adjusted[second_term_ixs] = df['Visits'].values[second_term_ixs] + df['Visits'].values[second_term_ixs]*0.04
    df['Visits'] = adjusted
    df.loc[df.Visits.isnull(), 'Visits'] = 0
    return df

if __name__ == '__main__':
    # predict for weekend
    train, test = get_raw_data()
    train, test = transform_data(train, test, periods=-49)
    preds_weekend = predict_using_median_weekend(train, test)
    # predict for other days
    train, test = get_raw_data()
    train, test = transform_data(train, test, periods=-8)
    preds_otherdays = predict_using_median_weirddays(train, test)

    # simple combine weekend traffic and otherdays traffic
    weight = 0.9
    preds_ensemble = preds_weekend*weight + preds_otherdays*(1-weight)
    test['Visits'] = preds_ensemble
    test = wiggle_preds(test)

    test[['Id','Visits']].to_csv('../submit/sub_mads_weight_{}.csv'.format(weight), index=False)
    print(test[['Id', 'Visits']].head(10))
    print(test[['Id', 'Visits']].tail(10))