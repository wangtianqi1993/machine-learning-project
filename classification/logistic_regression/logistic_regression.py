# !/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'wtq'


import time
from numpy import *
import matplotlib.pyplot as plt
from preprocessing.load_data_set import load_data_set

# from detector.logger import DetectorLogger
# logger = DetectorLogger()


def sigmoid(inx):
    return 1.0/(1+exp(-inx))


def train_logistic_regression(data_mat_in, class_labels, opts):
    """
    this is the gradient ascent algorithm for logistic regression
    :param data_mat_in:
    :param class_labels:
    :return:
    """
    start_time = time.time()
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    num_samples, num_features = shape(data_matrix)
    print 'num_samples', num_samples
    print 'num_features', num_features
    alpha = opts['alpha']
    max_cycles = opts['max_cycles']
    weights = ones((num_features, 1))
    # print "data_matrix", data_matrix
    # print "data_label", label_mat
    # print "mul", data_matrix*weights

    for k in range(max_cycles):
        # 最原始的梯度下降法将整个训练集合中的所有样本与weights相乘，叠加求和再计算误差
        # 体现在data_matrix*weights中，每迭代一次就要遍历整个样本集合，集合大时比较耗时，与之对应的是随机梯度下降（上升）
        # 改进的方法是一次仅用一个样本点（的回归误差）来更新回归系数。这个方法叫随机梯度下降算法。由于可以在新的样本到来的时候对
        # 分类器进行增量的更新（假设我们已经在数据库A上训练好一个分类器h了，那新来一个样本x。对非增量学习算法来说，我们需要把x和数据库A
        # 混在一起，组成新的数据库B，再重新训练新的分类器。但对增量学习算法，我们只需要用新样本x来更新已有分类器h的参数即可），所以它属于
        # 在线学习算法。与在线学习相对应，一次处理整个数据集的叫“批处理”
        if opts['optimize_type'] == 'graddescent':
            output = sigmoid(data_matrix*weights)
            error = (label_mat - output)
            # print "data_matrix...", data_matrix.transpose()
            print "error...", error
            # print "mul...", data_matrix.transpose()*error
            weights = weights + alpha * data_matrix.transpose()*error

        elif opts['optimize_type'] == 'stoc_graddescent':
            # 一次仅用一个样本点（的回归误差）来更新回归系数
            for i in range(num_samples):
                output = sigmoid(data_matrix[i, :]*weights)
                error = label_mat[i, 0] - output
                # print "data_matrix", data_matrix[i, :].transpose()
                print "error..", error
                # print "mul..", data_matrix[i, :].transpose()*error
                weights = weights + alpha*data_matrix[i, :].transpose()*error

        elif opts['optimize_type'] == 'smooth_stoc_graddescent':
            # 基于步长逐渐减小的优化梯度下降,且随机的选择样本
            data_index = range(num_samples)
            for i in range(num_samples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                rand_index = int(random.uniform(0, len(data_index)))
                output = sigmoid(data_matrix[data_index[rand_index], :]*weights)
                error = label_mat[data_index[rand_index], 0] - output
                weights = weights + alpha * data_matrix[data_index[rand_index], :].transpose()*error
                # during one interation, delete the optimized sample
                del(data_index[rand_index])

        else:
            raise NameError('Not support optimize method type!')

    print 'Congratulations, training complete! Took %fs!' % (time.time() - start_time)
    return weights


# test your trained Logistic Regression model given test set
def test_logistic_regression(weights, test_x, test_y):
    num_samples, num_features = shape(test_x)
    match_count = 0
    test_x = mat(test_x)
    test_y = mat(test_y).transpose()
    for i in xrange(num_samples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            match_count += 1
    accuracy = float(match_count) / num_samples
    return accuracy


# show your trained logistic regression model only available with 2-D data
def show_logistic_regression(weights, train_x, train_y):
    # notice: train_x, train_y is mat datatype
    train_x = mat(train_x)
    train_y = mat(train_y).transpose()
    num_samples, num_features = shape(train_x)
    if num_features != 3:
        print "Sorry I can not draw because the dimension of your data is not 2"
        return 1

    # dram all samples
    for i in xrange(num_samples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i ,2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i ,1], train_x[i ,2], 'ob')

    # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA() # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == "__main__":
    test_parameter = {}
    test_parameter['alpha'] = 0.001
    test_parameter['max_cycles'] = 100
    test_parameter['optimize_type'] = 'stoc_graddescent'
    train_data, train_label = load_data_set('test_data.txt')
    print train_data, train_label
    #w = train_logistic_regression(train_data, train_label, test_parameter)
    #show_logistic_regression(w, train_data, train_label)
    # logger.info(test_parameter['optimize_type'])
    # logger.info(w)
    # logger.info('test result')
    # logger.info(test_logistic_regression(w, train_data, train_label))
