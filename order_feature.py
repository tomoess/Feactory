# -*- coding: utf-8 -*-
'''
根据order表统计的消费特征
'''
import pandas as pd
from pandas import Series,DataFrame
import numpy as np


data_start = '2016-8'
data_end = '2016-10'

data_dir = 'data/'
dest_dir = 'feature/'

order_data = pd.read_csv(data_dir+'t_order.csv')
order_data['buy_time'] = pd.to_datetime(order_data['buy_time']) #将数据类型转换为日期类型
order_data = order_data.set_index('buy_time') # 将date设置为index
order_data = order_data[data_start:data_end]

#通过单价和数量算出总价
order_data['account'] = Series([order_data['price'][x]*order_data['qty'][x]  for x in range(0,len(order_data))])
grouped_uid_sum = order_data.groupby(by = ['uid'],as_index = False).sum()
price_sum = grouped_uid_sum['account']
discount_sum = grouped_uid_sum['discount']

#grouped_uid_cnt = order_data.groupby(by = ['uid'],as_index = False).count()
order_features = pd.DataFrame()
order_features['uid'] = order_data['uid']
order_features['account_sum'] = price_sum
order_features['disaccount_sum'] = discount_sum

order_features.to_csv(dest_dir+'order_features',index = False)