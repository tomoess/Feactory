# -*- coding: utf-8 -*-
"""
# 统一输入:csvfilename_list
#     一个list，
#           len(csvfilename_list) = 1时，propname_list为一个单层list，内存储一个或多个propname
#           len(csvfilename_list) > 1时，propname_list为一个双层list，每个单元为一个list，内存储一个或多个propname
# 统一输出:RTN_PROPFRAME_NC N = sum[len(propname_list)]
#     一个DataFrame，index为默认，一共N列，每列为抽取出的prop
"""

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import scipy.stats as sts  

# ------------------------------------------------------------------------------------
# 1:输出每个uid出现的总次数
def sub_func_get_uid_frecframe(DATAFRAME_2C):
    DATAFRAME_2C = pd.read_csv('data/t_loan.csv')[['uid','plannum']]
    DATAFRAME_2C['feature'] = Series(np.ones(len(DATAFRAME_2C)))
    RTN_PROPFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).count()[['uid','feature']]     
    return RTN_PROPFRAME_2C
# ------------------------------------------------------------------------------------
# 2:输出每个uid对应的所有prop的种类数
def sub_func_get_proptype_frame(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.drop_duplicates(['uid','prop'])
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).count()
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 3:输出每个uid对应的多个prop的简单求和
def sub_func_get_sum_propframe(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).count()
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 4:输出每个uid对应的所有prop的平均值
def sub_func_get_avg_propframe(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).count()
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 5:输出每个uid对应的所有prop的中位数
def sub_func_get_median_propframe(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).median()
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 6:输出每个uid对应的所有prop的众数，出现频率最高的
def sub_func_get_mode_propframe(DATAFRAME_2C):
    DATAFRAME_2C['cnt'] = Series(np.ones(len(DATAFRAME_2C)))
    uid_prop_cnt = DATAFRAME_2C.groupby(by = ['uid','prop'],as_index = False).count()#aggregating uid-prop,1 uid vs n prop    
    uid_prop_cnt_max = uid_prop_cnt.groupby(by = ['uid'],as_index = False).max()#aggregating for uid twice，get max freq of prop
    uid_prop_cnt_max =  pd.merge(uid_prop_cnt_max[['cnt','uid']],uid_prop_cnt,how='left')   
    uid_prop_cnt_max = pd.merge(uid_prop_cnt_max,uid_prop_cnt[['cnt','uid']],how='left').drop_duplicates()
    return duplicat_processor(uid_prop_cnt_max,[])
def duplicat_processor(uid_prop_tab,prop_priority_list = None):
    #此处需检查prop_priority_list是否存在重复，尚未实现   
    if(len(prop_priority_list)>0):
        priority_name =  list(set(prop_priority_list.keys()) ^ set(['prop']))[0]
        '''
        prop_priority_list.rename(columns={priority_name:'priority'}, inplace=True)
        prop_priority_list['dup'] = prop_priority_list['priority'].duplicated()
        prop_priority_list[].isin([True])
       
        assert (True not in prop_priority_list['priority'].duplicated()),'prop_priority_list has duplicates'
        '''
        uid_prop_priority_list =  pd.merge(uid_prop_tab,prop_priority_list,how='left')
        uid_prop_priority_max =  uid_prop_priority_list.groupby(by = ['uid'],as_index = False).max()[['uid','priority']]        
        no_duplicates_tab = pd.merge(uid_prop_priority_max[['uid','priority']],prop_priority_list,how='left')

    else:
        no_duplicates_tab = uid_prop_tab.drop_duplicates(['uid'])
    return no_duplicates_tab[['uid','prop']]
# ------------------------------------------------------------------------------------
# 7:输出每个uid对应的所有prop的上四分卫数
def sub_func_get_upquantile_propframe(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).quantile(0.75)
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 8:输出每个uid对应的所有prop的下四分卫数
def sub_func_get_bottomquantile_propframe(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).quantile(0.25)
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 9:输出每个uid对应的所有prop的累积乘积
def sub_func_get_cumprod_propframe(DATAFRAME_2C):
    return RTN_PROPFRAME_2C
# ------------------------------------------------------------------------------------
# 10:输出每个uid对应的所有prop在全局频度排序排名最高的prop值
def sub_func_get_topfrec_propframe(DATAFRAME_2C):
    prop_priority_list  = DATAFRAME_2C.groupby(by = ['prop'],as_index = False).count()[['uid','prop']]
    prop_priority_list.rename(columns={'uid':'priority'}, inplace=True)
    uid_prop_priority_list =  pd.merge(DATAFRAME_2C,prop_priority_list,how='left')
    uid_prop_priority_max =  uid_prop_priority_list.groupby(by = ['uid'],as_index = False).max()[['uid','priority']]        
    uid_prop_priority_max = pd.merge(uid_prop_priority_max[['uid','priority']],prop_priority_list,how='left').drop(['priority'],axis = 1)
    return uid_prop_priority_max.rename(columns={'prop':'feature'}, inplace=True)

# ------------------------------------------------------------------------------------
# 11:输出每个uid对应的所有prop中数值最小的
def sub_func_get_minnum_propframe(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).min()
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 12:输出每个uid对应的所有prop中数值最大的
def sub_func_get_maxnum_propframe(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).max()
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 13:输出每个uid对应的所有prop的极差
def sub_func_get_numrange_propframe(DATAFRAME_2C):
    RTN_PROPFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).max() - DATAFRAME_2C.groupby(by = ['uid'],as_index = False).min()
    return RTN_PROPFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 14:输出每个uid对应的所有prop的方差
def sub_func_get_var_propframe(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).var()
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 15:输出每个uid对应的所有prop的标准差
def sub_func_get_std_propframe(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).std()
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 16:输出每个uid对应的所有prop的四分卫差
def sub_func_get_quantilerange_propframe(DATAFRAME_2C):
    RTN_PROPFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).quantile(0.75) - DATAFRAME_2C.groupby(by = ['uid'],as_index = False).quantile(0.25)
    return RTN_PROPFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 17:输出每个uid对应的所有prop的离散系数(Coefficient of Variance) std/mean
def sub_func_get_coefficient_propframe(DATAFRAME_2C):
    RTN_PROPFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).std() - DATAFRAME_2C.groupby(by = ['uid'],as_index = False).mean()
    return RTN_PROPFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 18:输出每个uid对应的所有prop的一阶差分
def sub_func_get_diff_propframe(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).diff()
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 19:输出每个uid对应的所有prop的百分比数变化
def sub_func_get_pct_change_propframe(DATAFRAME_2C):
    DATAFRAME_2C = DATAFRAME_2C.groupby(by = ['uid'],as_index = False).pct_change()
    return DATAFRAME_2C.rename(columns={'prop':'feature'}, inplace=True)
# ------------------------------------------------------------------------------------
# 20:输出每个uid对应的所有prop不排序的变化量的标准差
def sub_func_get_diffstd_propframe(DATAFRAME_2C):
    return RTN_PROPFRAME_2C
# ------------------------------------------------------------------------------------
# 21:输出每个uid对应的所有prop排序后的变化量的标准差
def sub_func_get_sorteddiffstd_propframe(DATAFRAME_2C):
    return RTN_PROPFRAME_2C
# ------------------------------------------------------------------------------------
# 22:输出每个uid对应的所有prop的平均绝对离差(Mean Absolute Deviation)
def sub_func_get_mad_propframe(DATAFRAME_2C):
    return RTN_PROPFRAME_2C
# ------------------------------------------------------------------------------------
# 23:输出每个uid对应的所有prop中，出现频率最高的那种prop值
def sub_func_get_mostfrec_propframe(DATAFRAME_2C):
    return RTN_PROPFRAME_2C
# ------------------------------------------------------------------------------------
# 24:偏度(Skewness(三阶距))统计数据分布偏斜方向和程度的度量
def sub_func_get_skew_propframe(DATAFRAME_2C):
    return RTN_PROPFRAME_2C
# ------------------------------------------------------------------------------------
# 25:峰度系数(Kurtosis(四阶距))频数分布曲线顶端尖峭或扁平程度的指标
def sub_func_get_kurt_propframe(DATAFRAME_2C):
    return RTN_PROPFRAME_2C
