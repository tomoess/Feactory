# -*- coding: utf-8 -*-
'''
根据order表统计的消费特征
'''
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import time
from datetime import datetime

def get_user_feature(data_start,data_end,saveFileName,
                      data_dir = 'data/',dest_dir = 'features/'):

    

    user_data = pd.read_csv(data_dir+'t_user.csv')
    
    str2time = lambda x:time.mktime(time.strptime(x, "%Y-%m-%d"))
    active_time = [time.time() - str2time(x) for x in user_data['active_date']]

    user_data['active_time'] = Series(active_time)
    user_data = user_data[['uid','age','sex','limit','active_time']]
    user_data.to_csv(dest_dir+saveFileName,index = False)
    print('user_features file generated')

    
def test():
    data_start = '2016-8'
    data_end = '2016-10'

    get_user_feature(data_start,data_end,'user_features.csv')
