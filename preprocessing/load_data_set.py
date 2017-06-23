# !/usr/bin/env python
# -#- coding: utf-8 -*-
__author__ = 'wtq'


def load_data_set(path):
    """

    :param path:the file's path that you want to load
    :return:
    """
    data_mat = []
    label_mat = []
    fr = open(path)
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def load_data_nnet(path):
    read_data = []
    fr = open(path)
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat = [float(line_arr[0]), float(line_arr[1])]
        label_mat = [int(line_arr[2])]
        item = [data_mat, label_mat]
        read_data.append(item)
    return read_data

if __name__ == "__main__":
    print load_data_nnet("test_data.txt")

