# !/usr/bin/env python
# -*-coding: utf-8 -*-
__author__ = 'wtq'

import math
import random
from numpy import *
from preprocessing.load_data_set import load_data_nnet
from preprocessing.normalization.normailization import auto_norm
from preprocessing.normalization.normailization import auti_norm


def sigmoid(inx):
    return 1.0/(1+exp(-inx))

# def sigmoid(x):
#     return math.tanh(x)

# def dsigmoid(y):
#     return 1.0 - y**2


def dsigmoid(inx):
    return (float)(exp(-inx)/((1+exp(-inx))*(1+exp(-inx))))


def make_matrix(y, x, fill=0.0):
    m = []
    for i in range(y):
        m.append([fill]*x)
    return m


class Nauron:
    def __init__(self):
        pass


class NN:
    def __init__(self, numinput, numhidden, numoutput):
        """

        :param numinput: the number nodes of input layer
        :param numhidden: the number nodes of hidden layer
        :param numoutput: the number nodes of output layer
        :return:
        """
        self.numinput = numinput + 1
        self.numhidden = numhidden
        self.numoutput = numoutput

        self.inputact = [1.0] * self.numinput
        self.hiddenact = [1.0] * self.numhidden
        self.outputact = [1.0] * self.numoutput

        # 输出层上每个节点的诱导局部域, 在更新w时会用到
        self.output_in = [1] * self.numoutput
        # 隐藏层上每个节点的诱导局部域, 在更新w时会用到
        self.hidden_in = [1] * self.numhidden


        self.inputweights = make_matrix(self.numinput, self.numhidden)
        self.outputweights = make_matrix(self.numhidden, self.numoutput)

        # randomize weights随机生成两个权值矩阵
        for i in range(self.numinput):
            for j in range(self.numhidden):
                self.inputweights[i][j] = random.uniform(-0.2, 0.2)

        for j in range(self.numhidden):
            for k in range(self.numoutput):
                self.outputweights[j][k] = random.uniform(-0.2, 0.2)

        self.inputchange = make_matrix(self.numinput, self.numhidden)
        self.outputchange = make_matrix(self.numhidden, self.numoutput)

    def update(self, inputs):
        """Update network"""
        # 训练好网络后，可以将测试数据输入到update中，返回网络所预测的结果
        if len(inputs) != self.numinput - 1:
            raise ValueError('Wrong number of inputs, should have %i inputs.' % self.numinput)

        self.input_norm, self.input_range, self.input_min = auto_norm(inputs)

        # Activate input layers neurons (-1 ignore bias node)
        for i in range(self.numinput - 1):
            self.inputact[i] = self.input_norm[i]

        # Activate hidden layers neurous
        for h in range(self.numhidden):
            sum = 0.0
            # 下面这个for循环在计算第h个隐藏层节点的输入值，即为各个输入节点的输出值与该节点与隐藏层节间的权值之积再求和
            for i in range(self.numinput):
                sum = sum + self.inputact[i] * self.inputweights[i][h]
            # 隐藏层第h个节点的诱导局部域
            self.hidden_in[h] = sum
            # 隐藏层第h个节点的输出值
            self.hiddenact[h] = sigmoid(sum)

        # Activate output layers neurons
        for o in range(self.numoutput):
            sum = 0.0
            # 下面这个for循环在计算第h个输出层节点的输入值，即为各个隐藏曾节点的输出值与该节点与输出层节间的权值之积再求和

            for h in range(self.numhidden):
                sum = sum + self.hiddenact[h] * self.outputweights[h][o]

            # 输出层第o个节点的诱导局部域
            self.output_in[o] = sum
            # 输出层第h个节点的输出值
            self.outputact[o] = sigmoid(sum)
        # 输出值反归一化
        self.output = auti_norm(self.outputact, self.input_range, self.input_min)

        return self.output

    def back_propagate(self, targets, learningrate, momentum):
        """

        :param targets:  the ture output for the input
        :param learningrate: learn rate
        :param momentum:
        :return:
        """
        if len(targets) != self.numoutput:
            raise ValueError('Wrong number of target values.')

        # calculate error for output neurous
        output_deltas = [0.0] * self.numoutput
        for k in range(self.numoutput):
            error = targets[k] - self.output[k]
            output_deltas[k] = dsigmoid(self.output_in[k]) * error

        # calculate error for hidden neurons
        hidden_deltas = [0.0] * self.numhidden
        for j in range(self.numhidden):
            error = 0.0
            for k in range(self.numoutput):
                error = error + output_deltas[k] * self.outputweights[j][k]
            hidden_deltas[j] = dsigmoid(self.hidden_in[j]) * error

        # update output weights
        for j in range(self.numhidden):
            for k in range(self.numoutput):
                # 隐藏层第j个节点向输出层上各个节点输出时的值是相等的，为self.hiddenact[j
                change = output_deltas[k] * self.hiddenact[j]
                self.outputweights[j][k] += learningrate * change + momentum * self.outputchange[j][k]
                self.outputchange[j][k] = change

        # update input weights
        for i in range(self.numinput):
            for j in range(self.numhidden):
                change = hidden_deltas[j] * self.inputact[i]
                self.inputweights[i][j] += learningrate * change + momentum * self.inputchange[i][j]
                self.inputchange[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            # print ("target, output", targets[k], self.output[k])
            # 对于一个样本输出层各个节点均方误差累加，得到该样本的误差
            # error = error + 0.5 * (targets[k] - self.output[k]) ** 2
            error = error + (targets[k] - self.output[k])
        return error

    def train(self, patterns, iterations=100, learningrate=0.02, momentum=0.01):
        """
        training network a patterns
        :param patterns: the train sample
        :param iterations:
        :param learningrate:
        :param momentum:
        :return:
        """
        for i in range(iterations):
            error = 0.0

            for p in patterns:
                inputs = p[0]
                targets = p[1]

                # 采用随机梯度下降，对于一个特定样本用update()正向激活
                # 再用back_propagate()反向计算误差
                self.update(inputs)
                error = error + self.back_propagate(targets, learningrate, momentum)

            # 输出一次迭代的各个样本的训练误差之和
            print ('error %-.5f' % error)


def test():
    data = load_data_nnet("/home/wtq/develop/workspace/github/machine-learning-project/preprocessing/test_data.txt")
    network = NN(2, 4, 1)
    network.train(data)

if __name__ == '__main__':
    test()
