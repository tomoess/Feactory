# -*- coding:utf-8 -*-
from __future__ import print_function, division
import sklearn
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from enum import Enum
from tqdm import *
import math
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def Z_ScoreNormalization(list):
    avg = np.average(list)
    sigma = np.std(list)
    for i in range(len(list)):
        list[i] = (list[i] - avg) / sigma
    print(list)
    return list
def MaxMinNormalization(list):
    Min = np.min(list)
    Max = np.max(list)
    list = (list - Min) / (Max - Min);
    return list

def polyfit(list):
    x = range(len(list))
    y = list
    z = np.polyfit(x, y, 20)
    p = np.poly1d(z)
    # print(z)
    yvals = np.polyval(z, x)
    for i in range(len(list)):
        list[i] = yvals[i]
    return list


list1 = [10,11,12,13,14,15,16,17]
list2 = [3,2,1,4,5,6,8,7]
dict1 = dict()
dict2 = dict()
for i in range(len(list1)):
    dict1[i]=list1[i]
    dict2[i]=list2[i]

dict1 = {'a': 12, 'b': 11, 'c': 10, 'd': 13, 'e': 14, 'f': 15, 'g1': 17, 'h1': 16}
dict2 = {'a': 24, 'b': 22, 'c': 20, 'd': 23, 'e': 24, 'f': 25, 'g': 26, 'h': 27}



def KGrapher(rawdict1, rawdict2, funcnum=None):
    # normalize_dict_preprocess

    inter = dict.fromkeys([x for x in rawdict1 if x in rawdict2])
    dict1 = dict()
    dict2 = dict()
    for key in inter:
        dict1[key] = rawdict1[key]
        dict2[key] = rawdict2[key]

    print(dict1)
    print(dict2)

    print('*'*30)

    graphlen = len(dict1)
    bias = 1
    sortedduallist1 = sorted(dict1.items(), key=lambda d: d[1], reverse=False)
    rawvaluelist1 = []
    rawkeylist1 = []
    for k,v in sortedduallist1:
        rawvaluelist1.append(v)
        rawkeylist1.append(k)
    nlist1 = MaxMinNormalization(rawvaluelist1)


    sortedduallist2 = sorted(dict2.items(), key=lambda d: d[1], reverse=False)
    rawvaluelist2 = []
    rawkeylist2 = []
    for k, v in sortedduallist2:
        rawvaluelist2.append(v)
        rawkeylist2.append(k)
    nlist2 = MaxMinNormalization(rawvaluelist2)

    for i in range(graphlen):
        nlist2[i] += bias

    def findindex(x, targetlist, len):
        for i in range(len):
            if targetlist[i] == x:
                return i

    print(sortedduallist1)
    print(sortedduallist2)

    print(rawkeylist1)
    print(rawkeylist2)

    stringdict = dict()
    for i in rawkeylist1:
        stringdict[i]=findindex(i, rawkeylist2,graphlen)

    print(stringdict)

    print(nlist1)
    print(nlist2)


    def drawline(x1,y1,x2,y2):#TODO 根据偏移量改变alpha值
        if x2<x1:
            alpha = 0.2
        elif x2==x1:
            alpha = 0.5
        else:
            alpha = 0.8
        plt.plot([x1, x2], [y1, y2],
                 color='gray',
                 marker='o',
                 MarkerSize=0,
                 MarkerFaceColor='r',
                 MarkerEdgeColor='r',
                 linewidth=1.0,
                 alpha=alpha,
                 mec='r',
                 mfc='w')

    for i in range(graphlen):
        x = range(graphlen)
        y1 = nlist1
        y2 = nlist2
        plt.plot(x, y1,
                 color='red',
                 marker='o',
                 MarkerSize=0,
                 MarkerFaceColor='r',
                 MarkerEdgeColor='r',
                 linewidth=1.0,
                 alpha=0.5,
                 mec='r',
                 mfc='w')
        plt.plot(x, y2,
                 color='gray',
                 marker='o',
                 MarkerSize=0,
                 MarkerFaceColor='r',
                 MarkerEdgeColor='r',
                 linewidth=1.0,
                 alpha=0.5,
                 mec='r',
                 mfc='w')


    for i in range(graphlen):
        drawline(i,nlist1[i],
                 stringdict[rawkeylist1[i]],nlist2[stringdict[rawkeylist1[i]]])
    plt.legend()  # 让图例生效

    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("X")  # X轴标签
    plt.ylabel("Y")  # Y轴标签
    plt.title("Simple Plot")  # 标题

    plt.show()





KGrapher(dict1,dict2)
