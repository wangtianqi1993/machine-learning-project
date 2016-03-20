# !/user/bin/env python
# -*- coding: utf-8
__author__ = 'wtq'

import numpy as np
from numpy import *
import sys

def load_dataset(file_name, delim=' '):
    # fr = open(file_name)
    data_set = []
    with open(file_name, 'r') as fr:
        string_arr = [line.strip().split(delim) for line in fr.readlines()]
        # print string_arr
        # dat_arr = [map(change_float, line) for line in string_arr]
        for line in string_arr:
            temp = []
            for i in line:
                temp.append(float(i))
            data_set.append(temp)
    print data_set
    return mat(data_set)

# 根据要求的方差百分比，求出所需要的特征值的个数n(要降到的维度)
def percent2n(eigVals, percent):
    sortArray = np.sort(eigVals) #升序
    sortArray = sortArray[-1::-1] #逆转，降序
    arraySum = sum(sortArray)
    tmp = 0
    num = 0
    for i in sortArray:
        tmp += i
        num += 1
        if tmp >= arraySum*percent:
            return num
# 零均值化
def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal

def pca(dataMat, percent=0.8):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) #求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    n = percent2n(eigVals, percent)    #要达到percent的方差百分比，需要前n个特征向量
    eigValIndice = np.argsort(eigVals)   #对特征值从小到大排序
    n_gigValIndice = eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect = eigVects[:,n_gigValIndice]  #最大的n个特征值对应的特征向量
    lowDDataMat = newData*n_eigVect         #低维特征空间的数据
    reconMat = (lowDDataMat*n_eigVect.T)+meanVal  #重构数据
    return lowDDataMat, reconMat

if __name__ == '__main__':
    data_mat = load_dataset('/home/wtq/test.txt')
    print pca(data_mat)
