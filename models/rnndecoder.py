#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"

import tensorflow as tf
from collections import deque
import logging

log = logging.getLogger('RNNDecoder')


class HierarchicalGODecoder(object):
    BIOLOGICAL_PROCESS = 'GO:0008150'
    MOLECULAR_FUNCTION = 'GO:0003674'
    CELLULAR_COMPONENT = 'GO:0005575'
    FUNC_DICT = {'cc': CELLULAR_COMPONENT,
                 'mf': MOLECULAR_FUNCTION,
                 'bp': BIOLOGICAL_PROCESS}

    def __init__(self, inputlayer, num_negatives=10,
                 label_dimensions=128, learning_rate=0.001,
                 lstm_statesize=256, labelembedding=None):
        self.inputs = inputlayer
        self.learning_rate = learning_rate
        self.num_neg_samples = num_negatives
        self.label_dimensions = label_dimensions
        self.lstm_statesize = lstm_statesize
        self.labelembedding = labelembedding

    def init_variables(self, godag):
        self.ys_ = tf.placeholder(shape=[None, 5],
                                  dtype=tf.float32, name='y_out')

        self.labelemb = tf.get_variable('labelemb', initializer=self.labelembedding, trainable=False)
        self.threshold = tf.placeholder(shape=(1,), dtype=tf.float32, name='thres')
        self.negsamples = tf.placeholder(shape=[None, self.num_neg_samples])
        self.lstmcell = tf.contrib.rnn.BasicLSTMCell(self.lstm_statesize, activation=tf.nn.tanh, name='lstmcell')

        self.output_weights = tf.get_variable('rnn_outputW', shape=[self.lstm_statesize, self.label_dimensions])
        self.output_bias = tf.get_variable('rnnout_bias', shape=[self.label_dimensions])
        self.ytransform = tf.get_variable('ytransform', shape=[self.label_dimensions, self.label_dimensions],
                                          initializer=tf.initializer.identity)

    def build(self, godag):
        self.init_variables(godag)

        yemb = tf.embedding_lookup(self.labelemb, self.ys_, name='yemb')
        negemb = tf.embedding_lookup(self.labelemb, self.negsamples, name='negemb')
        rnnout, rnn_final_states = tf.nn.static_rnn(lstmcell, tf.zeros(shape=(tf.shape(self.yemb)[0].value, 5)),
                                                    initial_state=self.inputs,
                                                    sequence_length=5
                                                    )
        rflat = tf.reshape(rnnout, shape=[-1, self.lstm_statesize])

        # batchsize*5 x labeldim
        self.output = tf.nn.l2_normalize(tf.nn.softplus(tf.matmul(rflat,
                                                                  self.output_weights)
                                                        + self.output_bias,
                                                        name='yhat')
                                        )
        transformed_y = tf.nn.l2_normalize(tf.reshape(tf.matmul(yemb, self.ytransform),
                                                      shape=[-1, self.label_dimensions]),
                                           axis=1)
        # batch size*10 x labeldim
        transformed_negsamples = tf.nn.l2_normalize(tf.reshape(tf.matmul(negemb, self.ytransform),
                                                               shape=[-1, self.label_dimensions]),
                                                    axis=1)

        # batchsize *5 x 1
        cosinesim_pos = tf.sqrt(tf.reduce_sum(tf.multiply(self.output, transformed_y), axis=1))

        # batchsize *5 x batchsize*10
        cosinesim_neg = tf.sqrt(tf.matmul(self.output, transformed_negsamples.T))

        # batchsize *5 x 1
        min_neg_dist = tf.reduce_min(cosinesim_neg, axis=1)

        self.loss  = tf.reduce_mean(tf.exp(cosinesim_pos) / tf.exp(min_neg_dist))

        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train = self.optimizer.minimize(self.loss)

        self.predictions = self.make_prediction()
        self.precision, self.recall, self.f1score = calc_performance_metrics(self.ys_, self.prediction)
        return self

    def make_prediction(self):
        norm_labelemb = tf.nn.l2_normalize(tf.matmul(self.labelemb, self.ytransform), axis=1)
        distmat = tf.matmul(self.output, norm_labelemb.T)
        pred_labels = tf.argmin(distmat, axis=1)
