# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
import os
import datetime
import pandas as pd
import numpy as np

from mwviews.api import PageviewsClient



'''
Page example:

68834     U-Bahn_Zürich_de.wikipedia.org_desktop_all-agents
81856     Category:Images_from_the_New_York_Public_Libra...
87765              吹石一恵_ja.wikipedia.org_desktop_all-agents
30051            张继科_zh.wikipedia.org_all-access_all-agents
137595    Fibonacci-Folge_de.wikipedia.org_all-access_al...
60799               林夏薇_zh.wikipedia.org_desktop_all-agents
23571     Anne_Boleyn_fr.wikipedia.org_all-access_all-ag...
97590        Deus_Ex_ru.wikipedia.org_all-access_all-agents
65994     Kettenschifffahrt_auf_dem_Neckar_de.wikipedia....
72233     Compuesto_orgánico_es.wikipedia.org_desktop_al...
142714    Productos_notables_es.wikipedia.org_all-access...
80432     File:Dülmen,_Kirchspiel,_Oedlerteich_--_2016_-...
63714     ONE_PIECE角色列表_zh.wikipedia.org_desktop_all-agents
28525            张成泽_zh.wikipedia.org_all-access_all-agents
7462            Arménie_fr.wikipedia.org_desktop_all-agents
122666          永野芽郁_ja.wikipedia.org_all-access_all-agents
43428     Manual:Interface/Sidebar_www.mediawiki.org_des...
64869         14._April_de.wikipedia.org_desktop_all-agents
136013            政令指定都市_ja.wikipedia.org_all-access_spider
3561            我的女兒，琴四月_zh.wikipedia.org_all-access_spider
100353    ST_(рэпер)_ru.wikipedia.org_all-access_all-agents

81856 'Category:Images_from_the_New_York_Public_Library_commons.wikimedia.org_desktop_all-agents'
43428 'Manual:Interface/Sidebar_www.mediawiki.org_desktop_all-agents'
80432 'File:D\xc3\xbclmen,_Kirchspiel,_Oedlerteich_--_2016_--_1932.jpg_commons.wikimedia.org_mobile-web_all-agents'

we have website url
======
commons.wikimedia.org
www.mediawiki.org
zh.wikipedia.org
en.wikipedia.org
es.wikipedia.org
fr.wikipedia.org
de.wikipedia.org
ja.wikipedia.org
ru.wikipedia.org

we have access type
=====
all-access_spider
all-access_all-agents
desktop_all-agents
mobile-web_all-agents

Basically

desktop_all-agents + mobile-web_all-agents ~= all-access_all-agents
the visits of all-access_spider normally is lower than agents

'''
# parse the page string
# return article, project, access type, agent
def parse_page(page):
    matchObj = re.match(r'(.*)_([a-z][a-z].wikipedia.org)_(.*)_(spider|all-agents)',page)
    if matchObj:
        return matchObj.group(1), matchObj.group(2), matchObj.group(3), matchObj.group(4)
    
    matchObj = re.match(r'(.*)_(\w+.mediawiki.org)_(.*)_(spider|all-agents)',page)
    if matchObj:
        return matchObj.group(1), matchObj.group(2), matchObj.group(3), matchObj.group(4)
    
    matchObj = re.match(r'(.*)_(\w+.wikimedia.org)_(.*)_(spider|all-agents)',page)
    if matchObj:
        return matchObj.group(1), matchObj.group(2), matchObj.group(3), matchObj.group(4)
    
    print("Page can not parsed")
    print(page)
    return None, None, None, None

'''
mwviews article_views api

article_views(self, project, articles, access='all-access', agent='all-agents',
              granularity='daily',start=None, end=None)

project : str
    a wikimedia project such as en.wikipedia or commons.wikimedia
articles : list(str) or a simple str if asking for a single article
access : str
    access method (desktop, mobile-web, mobile-app, or by default, all-access)
agent : str
    user agent type (spider, user, bot, or by default, all-agents)
'''
exception_pages = list()

def get_page_views(page, pv_client, start_date=datetime.date(2017, 9, 1), end_date=None):
    global exception_pages
    # default is to get 2 days ago.
    if end_date == None:
        end_date = (datetime.datetime.now() - datetime.timedelta(days=2)).date()
    # parse the page
    articles, project, access, agent = parse_page(page)
    # fetch the views
    try:
        page_views = pv_client.article_views(project, articles, access, agent, start=start_date, end=end_date)
    except Exception:
        print(page)
        exception_pages.append(page)
        ds_page_views = pd.Series(np.nan, index=pd.date_range(start_date,end_date))
    else:
        ds_page_views = pd.DataFrame.from_dict(page_views).iloc[0]
        
    ds_page_views.index=ds_page_views.index.strftime('%Y-%m-%d')
    ds_page_views['Page']=page
    return ds_page_views


# load training data
train = pd.read_csv("../input/train_2.csv.zip",compression='zip')

START_DATE='20170901'
END_DATE='20170907'

# cache dir
cache_dir = '../cache/'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# client for downloading
pv_client = PageviewsClient()
extra_train = pd.DataFrame()

# split to 8 parts for downloading 
split_size = np.ceil(train.shape[0]/float(8))

# start to downloaded
for i in range(8):
    start_index = int(i * split_size)
    end_index = int(min((i+1)*split_size, train.shape[0]))
    extra_data = train.iloc[start_index:end_index].Page.apply(lambda x: get_page_views(x, pv_client, start_date=START_DATE, end_date=END_DATE))
    print("%d Pages downloaded"%extra_data.shape[0])
    # save to file
    extra_data.to_csv('../cache/extra_data_'+str(i)+'.csv.zip',index=False)
    extra_train = pd.concat([extra_train,extra_data])

    
filepath = '../input/extra_train_'+START_DATE+'_'+END_DATE+'.csv.zip'
extra_train.to_csv(filepath, index=False)

print("following pages downloading meet exception")
print(exception_pages)