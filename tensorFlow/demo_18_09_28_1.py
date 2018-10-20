#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib.pylab import plt

class Lstm(object):
    def __init__(self, taining_step=10000,n_input=28, n_steps=28, n_hidden=128, n_class=10,learning_rate=0.0001,
                 batch_size=100):
        self.input = n_input
        self.n_steps = n_steps
        self.n_hiden = n_hidden
        self.n_class = n_class
        self.lr = learning_rate
        self.taining_step = taining_step
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None, n_steps, n_input], 'x-input')
        self.y = tf.placeholder(tf.float32, [None, n_class], 'y-outPut')
        self.y_ = self.inference()
        self.mnist = input_data.read_data_sets('./mnist', one_hot=True)
    def inference(self):

        with tf.variable_scope('layer1'):
            x1 = tf.unstack(self.x, num=self.n_steps, axis=1, name='x-INPUT')
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hiden, forget_bias=1.0,)
            outputs, stats = tf.contrib.rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)
            y = tf.contrib.layers.fully_connected(outputs[-1], self.n_class, activation_fn=None)
            return y
    def train(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_, labels=tf.arg_max(self.y, 1))
        loss = tf.reduce_mean(cross_entropy)
        train_ = tf.train.AdamOptimizer(self.lr).minimize(loss)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.y_, 1), tf.arg_max(self.y, 1)),
                                          tf.float32))
        losses = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1, self.taining_step + 1):
                b_x, b_y = self.mnist.train.next_batch(self.batch_size)
                # b_x = np.reshape(b_x, [-1, 28, 28, 1])
                b_x = np.reshape(b_x, [-1, 28, 28])
                if i % 1000 == 0 or i == 1:
                    l, acc, t = sess.run([loss, accuracy, train_], feed_dict={self.x: b_x,
                                                                   self.y: b_y})
                    # sess.run(train_, feed_dict={self.x_input: b_x,
                    #                             self.y_: b_y})
                    print("{}第{}此迭代，此时的loss为：{},acc为{}".format(t, i, l, acc))
                else:
                    sess.run(train_, feed_dict={self.x: b_x,
                                                self.y: b_y})
                losses.append(sess.run(loss, feed_dict={self.x: b_x,
                                                self.y: b_y}))
            # if self.outPut_graph:
            writer = tf.summary.FileWriter("logs/", sess.graph)
            writer.close()
        return losses
if __name__ == "__main__":
    lstm = Lstm()
    loss = lstm.train()
    # loss = np.array(loss)
    plt.figure()
    plt.plot( loss)
    plt.show()