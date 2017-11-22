# -*- coding: utf-8 -*-
'''
把特征粗暴的塞进xgboost
'''
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import xgboost as xgb
from order_feature import get_order_feature
import sklearn.cross_validation as cv
data_dir = 'data/'
dest_dir = 'features/'

params={
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':12, # 构建树的深度，越大越容易过拟合
'min_child_weight':4,
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, # 如同学习率
'seed':1000,
'nthread':7,
'eval_metric': 'auc'
}
def get_uid_target_tab():

    loan_sum_data = pd.read_csv(data_dir+'t_loan_sum.csv')
    user_data = pd.read_csv(data_dir+'t_user.csv')
    loan_sum_data = loan_sum_data.drop('month',axis = 1 )

    loan_sum_all = pd.DataFrame()
    loan_sum_all['uid'] = user_data['uid']
    loan_sum_all = pd.merge(loan_sum_all,loan_sum_data,how='left')  
    loan_sum_all = loan_sum_all.fillna(0)
    return loan_sum_all
    
 

#   generating Features
print('generating Features...')
#get_order_feature('2016-8','2016-10','order_features.csv')

#  read in features and merge 
feature_dir = 'features/'
f_order = pd.read_csv(feature_dir + 'order_features.csv')
f_loan1 = pd.read_csv(feature_dir + 'loan_loan_amount.csv')
f_loan2 = pd.read_csv(feature_dir + 'loan_plannum.csv')
feature_tabs = [f_order,f_loan1,f_loan2]

data_all = get_uid_target_tab()
for feature_tab in feature_tabs:
    data_all = pd.merge(data_all,feature_tab,on='uid',how='left')  
#data split
train_set,val_set  = cv.train_test_split(data_all, test_size = 0.2,random_state=1)
X_train = train_set.drop('loan_sum',axis = 1)
Y_train = train_set['loan_sum']

X_val = val_set.drop('loan_sum',axis = 1)
Y_val = val_set['loan_sum']
xgb_train = xgb.DMatrix(X_train,label=Y_train)
xgb_val = xgb.DMatrix(X_val,label=Y_val)

# model training(xgb)

watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
XGBR = xgb.XGBRegressor(max_depth = 5,learning_rate=0.03,n_estimators=1600,reg_alpha=1,reg_lambda=0)
XGBR.fit(X_train.values,Y_train )
model.save_model('./model/xgb.model') # 用于存储训练出的模型
print("best best_ntree_limit",model.best_ntree_limit) 


'''  
#  train by xgb
XGBR = xgb.XGBRegressor(max_depth = 3,learning_rate=0.1,n_estimators=500)

#  pred by xgb
XGBR = xgb.XGBRegressor(max_depth = 3,learning_rate=0.1,n_estimators=500)
'''
