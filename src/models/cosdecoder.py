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
import sys
sys.path.append('../')
import ipdb
from utils import variable_summaries

log = logging.getLogger('root.RNNDecoder')


class GORNNDecoder(object):
    def __init__(self, inputlayer, labelembedding, num_negatives=10,
                 learning_rate=0.001,
                 lstm_statesize=256, numfuncs=5, trainlabelemb=False):
        self.inputs = inputlayer
        self.learning_rate = learning_rate
        self.num_neg_samples = num_negatives
        self.label_dimensions = labelembedding.shape[1]
        self.lstm_statesize = lstm_statesize
        self.labelembedding = labelembedding
        self.numfuncs = numfuncs
        self.trainlabelemb = trainlabelemb
        # self.GO_MAT = GO_MAT

    def init_variables(self):
        # First 5 leaf GO nodes  for a given sequence is only used.
        # size of ys_ is (batchsize x 5)
        self.ys_ = tf.placeholder(shape=[None, self.numfuncs],
                                  dtype=tf.int32, name='y_out')

        # this represents the label embedding, size (GO nodes x labelembeddingsize)
        self.labelemb = tf.get_variable('inputlabelemb', initializer=self.labelembedding, dtype=tf.float32,
                                        trainable=self.trainlabelemb)

        self.labelemb = tf.concat([tf.zeros((1, self.labelembedding.shape[1])), self.labelemb], axis=0, name='labelemb_concat')
        mask = tf.concat([[0], tf.ones(self.labelembedding.shape[0])], axis=0)

        ## to handle different tensorflow versions
        if hasattr(tf, 'initializers'):
            initializer = tf.initializers.random_uniform
        else:
            initializer = tf.random_uniform_initializer()

        self.output_weights = tf.get_variable('rnn_outputW', shape=[self.inputs.shape[-1], self.label_dimensions])
        self.output_bias = tf.get_variable('rnnout_bias', shape=[self.label_dimensions])
        self.ytransform = tf.get_variable('ytransform', shape=[self.label_dimensions, self.label_dimensions],
                                          initializer=initializer)

    def build(self):
        self.init_variables()

        ## batchsize x 5 x labelemb
        self.yemb = tf.nn.embedding_lookup(self.labelemb, self.ys_, name='yemb')

        log.info('input label embedding-{}'.format(self.yemb.get_shape()))

        # inputs is batch x sequenceEmbeddingSize
        tmp = tf.nn.softplus(tf.matmul(self.inputs, self.output_weights)
                                 + self.output_bias,
                                 name='yhat')
        try:
            self.output = tf.nn.l2_normalize(tmp,
                                             axis=1)
        except:
            self.output = tf.nn.l2_normalize(tmp,
                                             dim=1)

        log.info('final decoder out shape {}'.format(self.output.get_shape()))
        #tmp = tf.tensordot(self.yemb, self.ytransform, axes=[[2], [0]])
        tmp = self.yemb
        try:
            self.transformed_y = tf.nn.l2_normalize(tmp,
                                                    axis=2)
        except:
            self.transformed_y = tf.nn.l2_normalize(tmp,
                                                    dim=2)

        #ipdb.set_trace()
        variable_summaries(self.transformed_y)
        # batchsize *5 x 1
        self.cosinesim_pos = 1 - tf.abs(tf.reduce_sum(tf.multiply(tf.expand_dims(self.output, axis=1), self.transformed_y), axis=1))

        self.simmax = tf.reduce_max(self.cosinesim_pos)
        self.simmin = tf.reduce_min(self.cosinesim_pos)
        self.loss = tf.reduce_mean(tf.reduce_mean(self.cosinesim_pos, axis=1))
        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train = self.optimizer.minimize(self.loss)

        self.summary = tf.summary.merge_all()
        # self.predictions, self.precision, self.recall, self.f1 = self.make_prediction()
        self.predictions = tf.argmax(self.cosinesim_pos, axis=1, name='predictions') #self.make_prediction()
        return self
