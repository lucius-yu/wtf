# Web Traffic Forecasting

## Task

The training dataset consists of approximately 145k time series. Each of these time series represents a number of daily views of a different Wikipedia article, starting from July 1st, 2015 up until September 10th, 2017. The goal is to forecast the daily views between September 13th, 2017 and November 13th, 2017 for each article in the dataset. The name of the article as well as the type of traffic (all, mobile, desktop, spider) is given for each article.

The evaluation metric is symmetric mean absolute percentage error (SMAPE).
 
## Trials

1. The naive median solution from public kernel.
2. Imporved median solution with working day and non-working day median solution from public kernel.  It did show improvement
3. Median solution based on different windows, then mix with working-non-working day solution from public kernel. It did show improvement
4. Round the prediction to integer. It shows improvement on SMAPE objective, no matter which solution choosed. It did show improvement
5. Since working-non-working day solution better than naive median solution. It comes to using LightGBM or xgboost could be an approach. 
Based on cross-validation, GBDT shows improvements. At least improvement on prediction for 16 or 17 days at the beginning.
6. Using seasonal-decompose + lstm (one page one model), this naive solution worse than initial baseline. I am new to LSTM. So, deep neural
network approach might work. it is worth to check sjv's CNN solution https://github.com/sjvasquez/web-traffic-forecasting


## Solution

My final solution is based on LightGBM. 

### data preprocessing

Not so much, the number for page on each day could be from 0 , 1, 2 ... to a big number. It will make training hard. So take the log of the number of visits.
Meanwhile, to avoiding take the log of 0. then before take the log, +1 first. the formula is 
```
y = np.log(x + 1)
x = np.exp(y) - 1 # inverse
```

### training

Simple approach take objective 'poisson-regression', eval with 'mae' (mean absolute error). This approach works 
Then for other parameter, we could do some simple tunning

One day one model. So, total 62 models trained and used.

### cross validation

Take the 4 validation period

```
# train_start_date, train_end_date, valid_start_date, valid_end_date
valid_dates = [('2015-07-01', '2016-09-13', '2016-09-14', '2016-11-14'), 
               ('2015-07-01', '2017-03-14', '2017-03-15', '2017-05-15'),
               ('2015-07-01', '2016-10-11', '2016-10-12', '2016-12-12'),
               ('2015-07-01', '2017-05-09', '2017-05-10', '2017-07-10')]

```

LGB solution shows improvement for initial around 17 days (better result on each day for init 17 days) on all validation period
LGB solution shows improvement for all 62 days on last period. ( it might caused by more training features, each day's visit is a feature)
LGB solution shows the latest days visit number are more important. and some days around 1 years ago visit number are kind of important by print feature important 


### file description
src -- source code for final solution.
other -- some other tried solution codes
lstm -- lstm trial code

get_extra_data.py -- used to download wiki visit number information 

## Pity

from the Page, we in fact can get extra information. each page will be classified on Project (e.g. zh.wikipedia.com), Access (e.g. mobile-web, desktop, all-access). Agent (e.g. spider, agent)
Thoese information could be add to training and test data as categorial feature.

also, mobile-web + desktop ~ all-access relationship kept

I do not have time to try

## Issues

The final training data is only till 2017-09-10. i.e. 2 days gap. 
Also the NaN on the day 2017-09-10 is a little more than other days. 5000 vs 1500. This NaN could be reduced if we download data from wikipedia after some while
for example. the NaN on 2017-08-31 is more 20000 in the training data which published on begining of sep. but those NaN number on 2018-08-31 reduced a lot on the data publish on 2017-09-11


## finally

Explore the CNN solution and move to next
