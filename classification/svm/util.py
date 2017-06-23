# !/usr/bin/env python
# -*-coding: utf-8-*-
__author__ = 'wtq'

import random

def load_data_set(file_name):
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_jrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, h, l):
    """
    用于调整大于H或者小于L的alpha值
    :param aj:
    :param h:
    :param l:
    :return:
    """
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj

