# !/usr/bin/env python
# -*-coding: utf-8 -*-
__author__ = 'wtq'

from numpy import *


def standardization(x_mat, y_mat):
    """
    所有的特征减去各自的均值并除以方差,使每维度特征有相同的重要性
    :return:
    """
    x_mat = mat(x_mat)
    y_mat = mat(y_mat).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_means = mean(x_mat, 0)
    x_var = var(x_mat, 0)
    print 'var', x_var
    x_mat = (x_mat - x_means)/x_var
    return x_mat, y_mat

if __name__ == "__main__":
    x = [
        [1, 2, 3, 400],
        [2, 4, 7, 800],
        [9, 1, 3, 600]
    ]
    y = [1, 2, 3]
    print standardization(x, y)

