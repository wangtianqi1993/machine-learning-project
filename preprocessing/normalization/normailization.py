# !/usr/bin/env python
# -*-coding: utf-8-*-
__author__ = 'wtq'


def auto_norm(data_in):
    """
    normailization the input data
    :param data_in:
    :return:
    """
    normal_data = len(data_in)*[1.0]
    max_val = max(data_in)
    min_val = min(data_in)
    ranges = max_val - min_val

    for i in range(len(data_in)):
        normal_data[i] = float(float(data_in[i] - min_val) / ranges)
    return normal_data, ranges, min_val


def auti_norm(normal_data, ranges, min_val):
    """
    auti the normal_data and return
    :param normal_data:
    :param ranges:
    :param min_val:
    :return:
    """
    auti_data = len(normal_data)*[1.0]
    for i in range(len(normal_data)):
        auti_data[i] = normal_data[i]*ranges+min_val
    return auti_data
