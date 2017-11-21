
# -*- coding: utf-8 -*-
'''
把特征粗暴的塞进xgboost
'''
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import xgboost as xgb


import xgboost as xgb

XGBR = xgb.XGBRegressor(max_depth = 3,learning_rate=0.1,n_estimators=500)


#%%   readin data
feature_dir = 'features/'
f_order = pd.read_csv(feature_dir + 'order_features.csv')
f_loan1 = pd.read_csv(feature_dir + 'loan_loan_amount.csv')
f_loan2 = pd.read_csv(feature_dir + 'loan_plannum.csv')


