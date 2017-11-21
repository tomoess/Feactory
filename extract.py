import linecache
from tqdm import *

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

data_dir = '../data/'
print('Ready to roll.')
# t_click = pd.read_csv(data_dir+'t_click.csv')
# print('t_click done')
t_loan = pd.read_csv(data_dir+'t_loan.csv')
print('t_loan done')
# t_order = pd.read_csv(data_dir+'t_order.csv')
# print('t_order done')
# t_user = pd.read_csv(data_dir+'t_user.csv')
# print('t_user done')
# t_loan_sum = pd.read_csv(data_dir+'t_loan_sum.csv')
# print('t_loan_sum done')


# print(t_loan.head(5))


# uirlist = dict()

data_start_date_str = '2016-08-03'
data_end_date_str = '2016-09-05'


def restrictdate(x_time):
    if x_time <= data_end_date_str and x_time >= data_start_date_str:
        return True
    else:
        return False

def dictavg(dict):
    leng = len(dict.values())
    sum = 0.0
    # print(len)
    for x in dict.values():
        sum += x
    return sum/leng

# output module
# itemsumdict = dict()  指定uid在指定时间内对应的指定prop的总加和
# itemfrecdict = dict() 指定uid在指定时间内对应的指定prop所出现的次数
# itemavgprop = dict()  指定uid在指定时间内对应的指定prop的平均值

def outputoneprop(itemsumdict, itemfrecdict, itemavgprop, propname):
    import csv
    with open('loan_'+str(propname)+'.csv','w',newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        csvwriter.writerow(['uid', 'loan_'+str(propname)+'_amount_sum', 'loan_'+str(propname)+'_amount_frec', 'loan_'+str(propname)+'_avg_amount'])
        data = []
        for oneuid in itemsumdict.keys():
            data.append([str(oneuid),
                        str(itemsumdict[oneuid]),
                        str(itemfrecdict[oneuid]),
                        str(itemavgprop[oneuid])])
        csvwriter.writerows(data)
        datacsv.close()


cur_csv = t_loan

uidprop = 'uid'
timeprop = 'loan_time'
proplist = ['loan_amount', 'plannum']

for oneprop in proplist:

    listlen = len(cur_csv['uid'])
    print('len:'+str(listlen))

    itemsumdict = dict()
    itemfrecdict = dict()

    for i in tqdm(range(listlen)):
        if restrictdate(cur_csv[timeprop][i]):
            cur_uid = cur_csv[uidprop][i]
            if itemsumdict.get(cur_uid) == None:
                itemsumdict[cur_uid] = t_loan[oneprop][i]
                itemfrecdict[cur_uid] = 1
            else:
                itemsumdict[cur_uid] += t_loan[oneprop][i]
                itemfrecdict[cur_uid] += 1


    itemdictlen = len(itemsumdict)
    print('\nsumdictlen:'+str(itemdictlen))
    print('\navgitemlen:'+str(dictavg(itemfrecdict)))

    itemavgprop = dict()

    for onekey in tqdm(itemsumdict.keys()):
        itemavgprop[onekey] = itemsumdict[onekey]/itemfrecdict[onekey]

    print('\navgpropnum:'+str(dictavg(itemavgprop)))

    outputoneprop(itemsumdict, itemfrecdict, itemavgprop, oneprop)








