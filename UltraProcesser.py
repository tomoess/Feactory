from tqdm import *
import pandas as pd
import numpy as np



# func:
#       UltraProcesser
# input:
#     [str]:[csvfilename]
#     [str]:[input_uid_name]
#     [str]:[input_prop_name]
#     (uid的去重长度:[len_uid_unique])
#     (prop的去重长度:[len_prop_unique])
#     [int]:[func_num]
#         [1*len_uid_unique]
#             0:输出每个uid出现的总次数
#             1:输出每个uid对应的多个prop的简单求和
#             2:输出每个uid对应的所有prop的种类数
#             3:输出每个uid对应的所有prop中，出现频率最高的那种prop值（注意这里还没有加如果都最高按排名选！）
#             4:输出每个uid对应的在所有prop中频度排名最高的prop值
#             5:输出每个uid对应的所有prop的平均值
#             6:输出所有uid对应的prop按数值排序后拟合后处于的位置信息，>=拐点为1，<拐点为0，默认为polyfit，自由度为3
#         [1*len_prop_unique]
#             7:输出所有prop及其对应的频度
#  output2file:[default = False]
#     [True: write to a csv file with two content rows, return 1 if done else return 0]
#     [False: return a dict]
#  output:
#     if output2file == True, return 0 or 1 and write a csv file to disk
#     elif output2file == False, return a dict with contents

# 在这之前，要对数据进行是否有空运算和是否符合运算规则检查
# ------------------------------------------------------------------------------------
# 0:输出每个uid出现的总次数
def sub_func_uidfrec(uid_list, prop_list, totallen):
    itemfrecdict = dict()
    for i in tqdm(range(totallen)):
        cur_uid = uid_list[i]
        if itemfrecdict.get(cur_uid) == None:
            itemfrecdict[cur_uid] = 1
        else:
            itemfrecdict[cur_uid] += 1
    return itemfrecdict
# ------------------------------------------------------------------------------------
# 1:输出每个uid对应的多个prop的简单求和
#   这里的 uid_list, prop_list 都是已经经过restrict函数筛选过的，totallen <= maxtotallen
def sub_func_simplesumprop(uid_list, prop_list, totallen):
    itemsumdict = dict()
    for i in tqdm(range(totallen)):
            cur_uid = uid_list[i]
            if itemsumdict.get(cur_uid) == None:
                itemsumdict[cur_uid] = prop_list[i]
            else:
                itemsumdict[cur_uid] += prop_list[i]
    return itemsumdict
# ------------------------------------------------------------------------------------
# 1.5 功能函数 get item prop frec dict
def tool_func_getitempropfrecdict(uid_list, prop_list, totallen):
    itempropfrecdict = dict()
    for i in tqdm(range(totallen)):
        cur_uid = uid_list[i]
        if itempropfrecdict.get(cur_uid) == None:
            itempropfrecdict[cur_uid] = dict()
            itempropfrecdict[cur_uid][prop_list[i]] = 1
        else:
            if itempropfrecdict[cur_uid].get(prop_list[i]) == None:
                itempropfrecdict[cur_uid][prop_list[i]] = 1
            else:
                itempropfrecdict[cur_uid][prop_list[i]] += 1
    return itempropfrecdict
# ------------------------------------------------------------------------------------
# 2:输出每个uid对应的所有prop的种类数
def sub_func_getproptype(uid_list, prop_list, totallen):
    itempropfrecdict = tool_func_getitempropfrecdict(uid_list, prop_list, totallen)
    for onekey in itempropfrecdict:
        tmppropfrec = len(itempropfrecdict[onekey])
        itempropfrecdict[onekey] = tmppropfrec
    return itempropfrecdict
# ------------------------------------------------------------------------------------
# 3:输出每个uid对应的所有prop中，出现频率最高的那种prop值
def sub_func_getmostfrecprop(uid_list, prop_list, totallen):
    itempropfrecdict = tool_func_getitempropfrecdict(uid_list, prop_list, totallen)
    for onekey in itempropfrecdict:
        mostfrecproptime = sorted(itempropfrecdict[onekey], key=lambda x: itempropfrecdict[onekey][x])[-1]
        itempropfrecdict[onekey] = mostfrecproptime
    return itempropfrecdict
# ------------------------------------------------------------------------------------
# 4:输出每个uid对应的在所有prop中频度排名最高的prop值
def sub_func_getrankheadpropdict(uid_list, prop_list, totallen):
    itempropfrecdict = tool_func_getitempropfrecdict(uid_list, prop_list, totallen)
    propfrecdict = sub_func_getpropfrecdict(uid_list, prop_list, totallen)
    for onekey in itempropfrecdict:
        itempropfrecdict[onekey] = \
            tool_func_getheadpropbyrankdict(itempropfrecdict[onekey], propfrecdict)
    return itempropfrecdict
# ------------------------------------------------------------------------------------
# 5:输出每个uid对应的所有prop的平均值
def sub_func_getavgpropdict(uid_list, prop_list, totallen):
    itemproptypedict = sub_func_getproptype(uid_list, prop_list, totallen)
    itempropsumdict = sub_func_simplesumprop(uid_list, prop_list, totallen)
    # print(len(itemproptypedict))
    # print(len(itempropsumdict))
    for onekey in itempropsumdict:
        itempropsumdict[onekey] = itempropsumdict[onekey]/itemproptypedict[onekey]
    return itempropsumdict
# ------------------------------------------------------------------------------------
# 5.5 功能函数 归一化及拟合
def Z_ScoreNormalization(list):
    avg = np.average(list)
    sigma = np.std(list)
    for i in range(len(list)):
        list[i] = (list[i] - avg) / sigma
    print(list)
    return list

def polyfit(list):
    x = range(len(list))
    y = list
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    # print(z)
    yvals = np.polyval(z, x)
    for i in range(len(list)):
        list[i] = yvals[i]
    return list
# ------------------------------------------------------------------------------------
# 6:输出所有uid对应的prop按数值排序后拟合后处于的位置信息，>=拐点为1，<拐点为0，默认为polyfit，自由度为3
def sub_func_getgroupnumfromfitcurve(uid_list, prop_list, totallen):
    fited_normalized_prop_list = polyfit(Z_ScoreNormalization(prop_list))
    #TODO
    fakedict = dict()
    return fakedict
# ------------------------------------------------------------------------------------
# 7:输出所有prop及其对应的频度
def sub_func_getpropfrecdict(uid_list, prop_list, totallen):
    propfrecdict = dict()
    for i in tqdm(range(totallen)):
        if propfrecdict.get(prop_list[i]) == None:
            propfrecdict[prop_list[i]] = 1
        else:
            propfrecdict[prop_list[i]] += 1
    return propfrecdict

def tool_func_getheadpropbyrankdict(unknowndict, rankdict):
    for i in range(len(rankdict)):
        key = sorted(rankdict, key=lambda x: rankdict[x], reverse=True)[i]
        if unknowndict.get(key) != None:
           return key
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------





# TODO 暂时还没生效
def restrictrule(proptobechecked):
    # like startdate to enddate
    data_start_date_str = '2016-8'
    data_end_date_str = '2016-10'
    if proptobechecked <= data_end_date_str and proptobechecked >= data_start_date_str:
        return True
    else:
        return False
# TODO 抽离rule函数
def restrictfunc(csvfile, restrictprop_name, restrictrule=restrictrule):
    csvfile['new_'+str(restrictprop_name)] = pd.to_datetime(csvfile[restrictprop_name])
    csvfile = csvfile.set_index('new_'+str(restrictprop_name))
    data_start_date_str = '2016-8'
    data_end_date_str = '2016-10'
    return csvfile[data_start_date_str:data_end_date_str]


class configs(object):
    def __init__(self):
        self.data_dir = '../data/'
        self.feature_dir = '../features/'
        self.filename = 't_loan.csv'
    def __repr__(self):
        print('data_dir: '+str(self.data_dir))
        print('feature_dir: ' + str(self.feature_dir))

def UltraProcesser(configs, input_uid_name, input_prop_name, func_num, output2file=False, restrictfunc=None, input_restrictprop_name=None):
    func_list = {
        '0': sub_func_uidfrec,
        '1': sub_func_simplesumprop,
        '2': sub_func_getproptype,
        '3': sub_func_getmostfrecprop,
        '4': sub_func_getrankheadpropdict,
        '5': sub_func_getavgpropdict,
        '6': sub_func_getgroupnumfromfitcurve,
        '7': sub_func_getpropfrecdict
    }
    csvfile = pd.read_csv(configs.data_dir + configs.filename)
    if input_restrictprop_name != None and restrictfunc != None:
        csvfile = restrictfunc(csvfile,input_restrictprop_name)
    uid_list = csvfile[input_uid_name].tolist()
    prop_list = csvfile[input_prop_name].tolist()
    len_uid_list = len(uid_list)
    len_prop_list = len(prop_list)
    if len_uid_list != len_prop_list:
        print('ERR: list len mismatch')
        return
    else:
        totallen = len_uid_list
        rtndict = func_list[func_num](uid_list, prop_list, totallen)
        if output2file:
            import csv
            with open(str(configs.feature_dir) + str(configs.filename).replace('.csv','_') +
                              str(input_uid_name) + '_' + str(input_prop_name) + '_func' +
                              str(func_num) + '.csv', 'w', newline="") as datacsv:
                csvwriter = csv.writer(datacsv, dialect=("excel"))
                if func_num == '7':
                    csvwriter.writerow([str(input_prop_name), 'totalcount'])
                else:
                    csvwriter.writerow([str(input_uid_name), str(input_prop_name)+'_func' + str(func_num)])
                rtnlist = sorted(rtndict.items(), key=lambda item:item[0])
                csvwriter.writerows(rtnlist)
                datacsv.close()
        else:
            print(len(rtndict))



def main():
    print("Working in %s"%__name__)
    newconfig = configs()

    #demo
    UltraProcesser(newconfig, 'uid', 'plannum', '1', output2file=True)
    UltraProcesser(newconfig, 'uid', 'plannum', '2', output2file=True)
    UltraProcesser(newconfig, 'uid', 'plannum', '3', output2file=True)
    UltraProcesser(newconfig, 'uid', 'plannum', '4', output2file=True)
    UltraProcesser(newconfig, 'uid', 'plannum', '5', output2file=True)
    UltraProcesser(newconfig, 'uid', 'plannum', '7', output2file=True)

if __name__ == '__main__':
    main()




