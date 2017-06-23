# !/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'wtq'

import tensorflow as tf
from data_set.tensorflow_data import mnist_input_data

mnist = mnist_input_data.read_data_sets("MNIST_data/", one_hot=True)

def tensor_flow_test():
    """

    :return:
    """

    matrix1 = tf.constant([[3, 3]])
    matrix2 = tf.constant([[2], [2]])
    product = tf.matmul(matrix1, matrix2)

    sess = tf.Session()
    result = sess.run(product)
    print result
    sess.close()


def mnist_frist():
    """
    this is a simple mnist_test
    :return:
    """
    x = tf.placeholder("float", [None, 784])

    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, w) + b)

    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # start training model
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # test model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


if __name__ == '__main__':
    # tensor_flow_test()
    mnist_frist()
