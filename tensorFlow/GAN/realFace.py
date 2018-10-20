#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

class Gan(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets("../mnist/", one_hot=True)
        self.batch_size = 100
        self.width, height = 28, 28
        self.mnist_dim = 784
        self.random_dim = 10
    def G(self, x):
        reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
        with tf.variable_scope('generator', reuse=reuse):
            x = slim.fully_connected(x, 32, activation_fn=tf.nn.relu)
            x = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
            x = slim.fully_connected(x, self.mnist_dim, activation_fn=tf.nn.sigmoid)
        return x

    def D(self, x):
        reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
        with tf.variable_scope('discriminator', reuse=reuse):
            X = slim.fully_connected(X, 128, activation_fn=tf.nn.relu)
            X = slim.fully_connected(X, 32, activation_fn=tf.nn.relu)
            X = slim.fully_connected(X, 1, activation_fn=None)
        return X

    real_X = tf.placeholder(tf.float32, shape=[self.batch_size, self.mnist_dim])
    random_X = tf.placeholder(tf.float32, shape=[self.batch_size, self.random_dim])
    random_Y = G(random_X)
