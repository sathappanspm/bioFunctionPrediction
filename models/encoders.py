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
import ipdb
import logging
import sys
sys.path.append('../')
from utils import variable_summaries

log = logging.getLogger('encoder')

class CNNEncoder(object):
    def __init__(self, embedding_size=128, vocab_size=24,
                 stride=1, filternum=32, kernelsize=128, inputsize=2000,
                 poolstride=32, poolsize=64, outputsize=1024):
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.stride = stride
        self.filternum = filternum
        self.kernelsize = kernelsize
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.outputsize = outputsize
        self.inputsize = inputsize
        self.outputs = None

    def init_variables(self):
        self.xs_ = tf.placeholder(shape=[None, self.inputsize], dtype=tf.int32, name='x_in')

        mask = tf.concat([[0], tf.ones(self.vocab_size - 1)], axis=0)
        # input activation variables
        self.emb = tf.get_variable('emb', [self.vocab_size, self.embedding_size],
                                   dtype=tf.float32, initializer=tf.initializers.random_uniform)

        self.emb = tf.reshape(mask, shape=[-1, 1]) * self.emb
        ## cnn kernel takes in shape [size,  (input channels, output channels)]
        self.cnnkernel = tf.get_variable('kernel', [self.kernelsize, self.embedding_size, self.filternum],
                                         dtype=tf.float32)


    def build(self):
        self.init_variables()

        self.cnn_inputs = tf.nn.dropout(tf.nn.embedding_lookup(self.emb, self.xs_, name='cnn_in'), 0.2)
        self.cnnout = tf.nn.elu(tf.nn.conv1d(self.cnn_inputs, self.cnnkernel, 1,
                                              'VALID', data_format='NWC', name='cnn1'))

        # log.info('shape-{}'.format(str(tf.shape(self.cnnout))))
        self.maxpool = tf.layers.max_pooling1d(self.cnnout, self.poolsize,
                                              self.poolstride, name='maxpool1')

        log.info('shape_cnnout-{}'.format(str(self.maxpool.get_shape())))
        # self.maxpool = tf.reshape(self.maxpool, shape=[tf.shape(self.maxpool)[0], -1])
        self.maxpool = tf.layers.Flatten()(self.maxpool)

        log.info('shape_maxpool-{}'.format(str((self.maxpool.get_shape()))))

        variable_summaries(self.maxpool)
        self.fcweights = tf.get_variable('fc1', shape=[self.maxpool.shape[1],
                                                       self.outputsize],
                                         dtype=tf.float32)
        self.fcbias = tf.get_variable('fcbias', shape=[self.outputsize])
        self.outputs = tf.nn.elu(tf.matmul(self.maxpool, self.fcweights) + self.fcbias, name='enc_out')
        variable_summaries(self.outputs)
        log.info('shape_encoderout-{}'.format(str((self.outputs.get_shape()))))
        return self


