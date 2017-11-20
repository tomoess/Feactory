# -*- coding: utf-8 -*-
'''
按月份统计loan表特征
目前仅有每月总借款额度特征
'''
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

data_dir = 'data/'
loan_data = pd.read_csv(data_dir+'t_loan.csv')
date_column = loan_data['loan_time']#造出一个统计月份的列

month_column =Series([ pd.to_datetime(x).month for x in date_column])
loan_data['month'] = month_column

grouped = loan_data.groupby(by = ['uid','month'],as_index = False).sum()
