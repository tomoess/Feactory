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
from sklearn.model_selection import cross_val_score
data_dir = 'data/'
dest_dir = 'features/'

model_xgb = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=3000,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=0)


def rmse_cv2(model,X,Y):
    rmse= np.sqrt(-cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv = 5))
    return(rmse)


def get_uid_target_tab():

    loan_sum_data = pd.read_csv(data_dir+'t_loan_sum.csv')
    user_data = pd.read_csv(data_dir+'t_user.csv')
    loan_sum_data = loan_sum_data.drop('month',axis = 1 )

    loan_sum_all = pd.DataFrame()
    loan_sum_all['uid'] = user_data['uid']
    loan_sum_all = pd.merge(loan_sum_all,loan_sum_data,how='left')  
    loan_sum_all = loan_sum_all.fillna(0)
    return loan_sum_all
    
 


def get_feature_tabs(mode = 'train'):
    print('generating Features...')
    if(mode == 'train'):
        start_date = '2016-8'
        end_date = '2016-10'
    else:
        start_date = '2016-9'
        end_date = '2016-11'
    mode = mode+'_'
    get_order_feature(start_date,end_date,mode+'order_features.csv')
    
    #  read in features and merge 
    feature_dir = 'features/'
    f_order = pd.read_csv(feature_dir + mode+'order_features.csv')
    #f_loan1 = pd.read_csv(feature_dir + 'loan_loan_amount.csv')
    #f_loan2 = pd.read_csv(feature_dir + 'loan_plannum.csv')
    feature_tabs = [f_order]
    return feature_tabs

#%%   generating train Features...
data_all = get_uid_target_tab()
train_feature_tabs = get_feature_tabs(mode = 'train')
for feature_tab in train_feature_tabs:
    data_all = pd.merge(data_all,feature_tab,on='uid',how='left')  
data_all = data_all.fillna(0)

#%% processing features
#连续值属性
featureConCols = ['active_time']
#离散值属性
featureCatCols = []
#%% data split and train
X = data_all.drop(['loan_sum','uid'],axis = 1)
Y = data_all['loan_sum']

X_train, X_val, Y_train, Y_val =  cv.train_test_split(X, Y, test_size=0.2, random_state=33)  
# model training(xgb)

print('training ...')
model_xgb.fit(X_train,Y_train,eval_metric='rmse', verbose = True, 
              eval_set = [(X_val, Y_val)],early_stopping_rounds=100 )


#%% predicting for submition
test_data = pd.DataFrame()
test_data['uid'] = data_all['uid']
test_feature_tabs = get_feature_tabs(mode = 'test')
for feature_tab in test_feature_tabs:
    test_data = pd.merge(test_data,feature_tab,on='uid',how='left')  

pred_test = model_xgb.predict(test_data.drop('uid',axis = 1).value)
import time

time_str=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))[5:]
savedData=pd.DataFrame({'id':test_data['uid'],'prob':pred_test})
test_data.to_csv(time_str+'submit.csv',index = False,header=False)    
