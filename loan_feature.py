# -*- coding: utf-8 -*-
'''
按月份统计loan表特征
目前仅有每月总借款额度特征
'''
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import math
data_dir = 'data/'
dest_dir = 'features/'


def get_loan_feature(data_start,data_end,saveFileName,
                      data_dir = 'data/',dest_dir = 'features/'):

    loan_data = pd.read_csv(data_dir+'t_loan.csv')
    #print('data loaded')
    
    loan_data['buy_time'] = pd.to_datetime(loan_data['loan_time']) #将数据类型转换为日期类型
    loan_data = loan_data.set_index('buy_time') # 将date设置为index
    loan_data = loan_data[data_start:data_end]
    
    
    #print('figuring account...')
    decode = lambda x:math.ceil(5**x -1)
    loan_data['loan_amout'] = [decode(a) for a in loan_data['loan_amount']]#计算脱敏前数据
 
    loan_data_sum = loan_data.groupby(by = ['uid'],as_index = False).sum()
    loan_data_count = loan_data.groupby(by = ['uid'],as_index = False).count()
    
    loan_features_all = pd.DataFrame()
    loan_features_all['uid'] =  loan_data_sum['uid']
    loan_features_all['loan_amout_sum'] = loan_data_sum['loan_amout']#借贷总额度
    #每个人的总分期数/每个人的借贷次数
    loan_features_all['loan_planum_habit'] = loan_data_sum['plannum']/loan_data_count['plannum']
    #sum(loan_amount/plannum)/n * 1/BZC（loan_amount/plannum）(置信度稳定性) = 少量多次否
    loan_features_all['loan_plannum_sum'] = loan_data_sum['plannum']#三个月内借款总次数
    #*标准差（单位时间内借款次数）

    loan_features_all.to_csv(dest_dir+saveFileName,index = False)
    print('order_features file generated')
    
def test():
    data_start = '2016-8'
    data_end = '2016-10'

    get_loan_feature(data_start,data_end,'order_features.csv')
