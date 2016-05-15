# !/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'wtq'

import string
import matplotlib.pyplot as plt
import numpy as np


def demo1():

    with open("/home/wtq/develop/workspace/github/machine-learning-project/preprocessing/test_data.txt", "r") as file:
        lines_list = file.readlines()
        lines_list = [line.strip().split() for line in lines_list]

        years = [float(x[0]) for x in lines_list]
        print years
        price = [float(x[1]) for x in lines_list]
        print price
        plt.plot(years, price, 'b*')
        plt.plot(years, price, 'r')
        plt.ylabel("housing average price(*2000 yuan)")
        plt.ylim(0, 15)
        plt.title('line_regression & gradient decrease')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    demo1()
