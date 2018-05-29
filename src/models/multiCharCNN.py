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
# import ipdb
import logging
from collections import deque
from tensorflow.contrib import rnn
import sys

sys.path.append('../')
from utils import calc_performance_metrics, full_eval_matrix

log = logging.getLogger('multiCNN')


class MultiCharCNN(object):
    def __init__(self, embedding_size=64, vocab_size=24,
                 stride=1, charfilter=32,
                 inputsize=2000, poolstride=16,
                 poolsize=16, with_dilation=False,
                 pretrained_embedding=None):
        self.vocab_size = vocab_size
        self.charfilter = charfilter
        self.inputsize = inputsize
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.outputs = None
        self.with_dilation = with_dilation
        self.pretrained_embedding = pretrained_embedding
        if self.pretrained_embedding is not None:
            self.embedding_size = self.pretrained_embedding.shape[1]
        else:
            self.embedding_size = embedding_size

    def init_variables(self):
        self.xs_ = tf.placeholder(shape=[None, self.inputsize], dtype=tf.int32, name='x_in')

        mask = tf.concat([[0], tf.ones(self.vocab_size - 1)], axis=0)

        if hasattr(tf, 'initializers'):
            initializer = tf.initializers.random_uniform
        else:
            initializer = tf.random_uniform_initializer()

        # input activation variables
        if self.pretrained_embedding is None:
            self.emb = tf.get_variable('emb', [self.vocab_size, self.embedding_size],
                                       dtype=tf.float32, initializer=initializer)
        else:
            log.info('model uses pretrained embedding-{}, {}'.format(self.pretrained_embedding.shape,
                                                                     self.pretrained_embedding.dtype))
            self.emb = tf.get_variable('emb', dtype=tf.float32, initializer=self.pretrained_embedding,
                                       trainable=False)

        self.emb = tf.reshape(mask, shape=[-1, 1]) * self.emb

        # size 3 k-mer kernel
        self.cnnkernel3 = tf.get_variable('kernel3', [3, self.embedding_size, self.charfilter],
                                    dtype=tf.float32)

        # size 5 k-mer kernel
        self.cnnkernel5 = tf.get_variable('kernel5', [5, self.embedding_size, self.charfilter],
                                    dtype=tf.float32)

        # size 7 k-mer kernel
        self.cnnkernel7 = tf.get_variable('kernel7', [7, self.embedding_size, self.charfilter],
                                    dtype=tf.float32)

    def build_biRNN(self, inputx):
        lstm_fw_cell = rnn.BasicLSTMCell(self.rnnhidden, forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(self.rnnhidden, forget_bias=1.0)
        self.birnn, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputx, dtype=tf.float32)
        return birnn

    def build(self):
        self.init_variables()
        #cnn_inputs = tf.nn.dropout(tf.nn.embedding_lookup(self.emb, self.xs_, name='cnn_in'), 0.2)
        cnn_inputs = tf.nn.embedding_lookup(self.emb, self.xs_, name='cnn_in')

        # 1998 x 32
        cnn3out = tf.nn.tanh(tf.nn.conv1d(cnn_inputs, self.cnnkernel3, 1,
                                               'VALID', data_format='NHWC', name='cnn3'))

        # 1996 x 32
        cnn5out = tf.nn.tanh(tf.nn.conv1d(cnn_inputs, self.cnnkernel5, 1,
                                               'VALID', data_format='NHWC', name='cnn5'))

        # 1994 x 32
        cnn7out = tf.nn.tanh(tf.nn.conv1d(cnn_inputs, self.cnnkernel7, 1,
                                               'VALID', data_format='NHWC', name='cnn7'))

        flatten = False
        if self.with_dilation:
            log.info('using dilation')
            cnn3out = tf.nn.tanh(tf.layers.conv1d(cnn3out, self.charfilter,
                                         64, padding='VALID',
                                         dilation_rate=2))

            cnn5out = tf.nn.tanh(tf.layers.conv1d(cnn5out, self.charfilter,
                                         64, padding='VALID',
                                         dilation_rate=4))

            cnn7out = tf.nn.tanh(tf.layers.conv1d(cnn7out, self.charfilter,
                                         64, padding='VALID',
                                         dilation_rate=6))
            flatten = True

        log.info('cnn3-{}, cnn5-{}, cnn7-{}'.format(cnn3out.get_shape(), cnn5out.get_shape(), cnn7out.get_shape()))

        m3 = tf.layers.max_pooling1d(cnn3out, self.poolsize, self.poolstride, padding='SAME')
        m5 = tf.layers.max_pooling1d(cnn5out, self.poolsize, self.poolstride, padding='SAME')
        m7 = tf.layers.max_pooling1d(cnn7out, self.poolsize, self.poolstride, padding='SAME')

        self.outputs = tf.concat([m3, m5, m7], axis=1)

        if flatten is True:
            self.outputs = tf.contrib.layers.flatten(self.outputs)

        log.info('shape_encoderout-{}'.format(str((self.outputs.get_shape()))))
        return self


class HierarchicalGODecoder(object):
    BIOLOGICAL_PROCESS = 'GO:0008150'
    MOLECULAR_FUNCTION = 'GO:0003674'
    CELLULAR_COMPONENT = 'GO:0005575'
    FUNC_DICT = {'cc': CELLULAR_COMPONENT,
                 'mf': MOLECULAR_FUNCTION,
                 'bp': BIOLOGICAL_PROCESS}

    def __init__(self, funcs, inputlayer, root='mf',
                 lossfunc=tf.nn.sigmoid_cross_entropy_with_logits,
                 learning_rate=0.001):
        self.root = HierarchicalGODecoder.FUNC_DICT.get(root, '')
        self.inputs = inputlayer
        self.lossfunc = lossfunc
        self.learning_rate = learning_rate
        self.funcs = funcs
        log.info('decoder is set to predict for {}'.format(len(funcs)))

    def get_node_func(self, node):
        name = node.split(':')[1]
        attnprob = tf.sigmoid(tf.matmul(tf.reshape(self.inputs,
                                                   shape=(-1, self.inputs.get_shape()[2])),
                                        self.attn))

        attnmpool = (self.inputs * tf.reshape(attnprob, shape=(-1, self.inputs.get_shape()[1], 1)))
        context = tf.reduce_sum(attnmpool, axis=1)
        var = tf.get_variable(name, shape=[context.get_shape()[1], 1], dtype=tf.float32)
        bias = tf.get_variable('{}_b'.format(name), shape=[1], dtype=tf.float32)
        return tf.sigmoid(tf.matmul(context, var) + bias, name='{}_out'.format(name))

    def init_variables(self, godag):
        self.ys_ = tf.placeholder(shape=[None, len(godag.GOIDS)],
                                  dtype=tf.float32, name='y_out')
        self.attn = tf.get_variable('attnwghts', shape=(self.inputs.get_shape()[2], 1))
        self.threshold = tf.placeholder(dtype=tf.float32, shape=(1,), name='thres')
        self.layers = {}
        queue = deque()
        funcset = set(self.funcs)
        for node in self.funcs:
            self.layers[node] = self.get_node_func(node)
            if not funcset.intersection(godag.isagraph.successors(node)):
                parents = [p for p in list(funcset.intersection(godag.isagraph.predecessors(node)))
                           if p not in queue]
                queue.extend(parents)

        visited = set(queue)
        log.info('created node for all functions')

        while len(queue) > 0:
            node = queue.popleft()
            out = self.layers[node]
            for ch_id in godag.isagraph.successors(node):
                if ch_id in self.layers:
                    out = tf.maximum(out, self.layers[ch_id])

            self.layers[node] = out
            parents = [p for p in list(funcset.intersection(godag.isagraph.predecessors(node)))
                       if p not in visited]
            visited.update(parents)
            queue.extend(parents)

        log.info('done with max merge')
        output = []
        for fn in self.funcs:
            output.append(self.layers[fn])

        self.output = tf.concat(output, axis=1)
        log.info('shape of output is {}'.format(self.output.shape))
        return

    def build(self, godag):
        self.init_variables(godag)
        clippedout = tf.clip_by_value(self.output, tf.constant(1e-7), 1 - tf.constant(1e-7))
        logits = tf.log(clippedout / (1 - clippedout))
        self.loss = tf.reduce_mean(self.lossfunc(labels=self.ys_[:, :len(self.funcs)], logits=logits))

        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train = self.optimizer.minimize(self.loss)

        self.prediction = tf.concat([self.output,
                                     tf.zeros((tf.shape(self.output)[0],
                                              len(godag.GOIDS) - len(self.funcs)))], axis=1, name='prediction')

        self.precision, self.recall, self.f1score = calc_performance_metrics(self.ys_, self.prediction, self.threshold)
        tf.summary.scalar('f1', self.f1score)
        self.summary = tf.summary.merge_all()
        return self

