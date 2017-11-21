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
dest_dir = 'features/'

order_data = pd.read_csv(data_dir+'t_order.csv')
print('data loaded')

order_data['buy_time'] = pd.to_datetime(order_data['buy_time']) #将数据类型转换为日期类型
order_data = order_data.set_index('buy_time') # 将date设置为index
order_data = order_data[data_start:data_end]


print('figuring account...')
order_data['account'] = order_data['price'] * order_data['qty']#通过单价和数量算出总价。
#现在的价格没有可加性，后续应加上幂函数变换

print('figuring groupby_sum...')
grouped_uid_sum = order_data.groupby(by = ['uid'],as_index = False).sum()
account_sum = grouped_uid_sum['account']
discount_sum = grouped_uid_sum['discount']

#grouped_uid_cnt = order_data.groupby(by = ['uid'],as_index = False).count()
order_features = pd.DataFrame()
order_features['uid'] = grouped_uid_sum['uid']
order_features['account_sum'] = account_sum
order_features['disaccount_sum'] = discount_sum

print('saving...')
order_features.to_csv(dest_dir+'order_features.csv',index = False)
print('order_features.csv generated')
