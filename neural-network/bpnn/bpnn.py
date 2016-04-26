# !/usr/bin/env python
# -*-coding: utf-8 -*-
__author__ = 'wtq'

import math
import random
from preprocessing.normalization.normailization import auto_norm
from preprocessing.normalization.normailization import auti_norm


def sigmoid(x):
    return math.tanh(x)


def dsigmoid(y):
    return 1.0 - y**2


def make_matrix(y, x, fill=0.0):
    m = []
    for i in range(y):
        m.append([fill]*x)
    return m


class Nauron:
    def __init__(self):
        pass
