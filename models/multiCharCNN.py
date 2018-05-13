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
from tensorflow.python.client import device_lib

sys.path.append('../')
from utils import calc_performance_metrics, full_eval_matrix

log = logging.getLogger('multiCNN')



def calc_performance_metrics(labels, predictions, threshold=0.35):
    labels = tf.cast(labels, tf.bool)
    predictions = predictions > threshold
    tp = tf.reduce_sum(tf.cast(tf.logical_and(labels, predictions), tf.float32), axis=1)
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(labels), predictions), tf.float32), axis=1)
    fn = tf.reduce_sum(tf.cast(tf.logical_and(labels, tf.logical_not(predictions)), tf.float32), axis=1)
    precision = tp / (tp + fp + tf.constant(1e-7))
    recall = tp / (tp + fn + tf.constant(1e-7))
    f1 = 2 * (precision * recall) / (precision + recall + tf.constant(1e-7))
    return (
        tf.reduce_mean(precision, name='precision'),
        tf.reduce_mean(recall, name='recall'),
        tf.reduce_mean(f1, name='f1')
        )


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    log.info('identified {} gpus'.format(len(gpus)))
    return gpus


class MultiCharCNN(object):
    def __init__(self, embedding_size=64, vocab_size=24,
                 stride=1, charfilter=32,
                 inputsize=2000,
                 poolstride=16, poolsize=16):
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.charfilter = charfilter
        self.inputsize = inputsize
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.outputs = None
        self.gpus = get_available_gpus()

    def init_variables(self):
        self.xs_ = tf.placeholder(shape=[None, self.inputsize], dtype=tf.int32, name='x_in')

        mask = tf.concat([[0], tf.ones(self.vocab_size - 1)], axis=0)
        # input activation variables
        self.emb = tf.get_variable('emb', [self.vocab_size, self.embedding_size],
                                   dtype=tf.float32, initializer=tf.initializers.random_uniform)

        self.emb = tf.reshape(mask, shape=[-1, 1]) * self.emb

    def build_biRNN(self, inputx):
        lstm_fw_cell = rnn.BasicLSTMCell(self.rnnhidden, forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(self.rnnhidden, forget_bias=1.0)
        self.birnn, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputx, dtype=tf.float32)
        return birnn

    def build(self):
        self.init_variables()
        cnn_inputs = tf.nn.dropout(tf.nn.embedding_lookup(self.emb, self.xs_, name='cnn_in'), 0.2)
        alloutput = []
        # size 3 k-mer kernel
        with tf.device(self.gpus[len(self.gpus) % 1]):
            cnnkernel3 = tf.get_variable('kernel3', [3, self.embedding_size, self.charfilter],
                                              dtype=tf.float32)
            cnn3out = tf.nn.relu(tf.nn.conv1d(cnn_inputs, cnnkernel3, 1,
                                               'VALID', data_format='NWC', name='cnn3'))
            m3 = tf.layers.max_pooling1d(cnn3out, self.poolsize, self.poolstride, padding='SAME')
            alloutput.append(m3)

        # size 5 k-mer kernel
        with tf.device(self.gpus[len(self.gpus) % 2]):
            cnnkernel5 = tf.get_variable('kernel5', [5, self.embedding_size, self.charfilter],
                                         dtype=tf.float32)
            cnn5out = tf.nn.relu(tf.nn.conv1d(cnn_inputs, cnnkernel5, 1,
                                              'VALID', data_format='NWC', name='cnn5'))
            m5 = tf.layers.max_pooling1d(cnn5out, self.poolsize, self.poolstride, padding='SAME')
            alloutput.append(m5)

        # size 7 k-mer kernel
        with tf.device(self.gpus[len(self.gpus) % 3]):
            cnnkernel7 = tf.get_variable('kernel7', [7, self.embedding_size, self.charfilter],
                                              dtype=tf.float32)
            cnn7out = tf.nn.relu(tf.nn.conv1d(cnn_inputs, cnnkernel7, 1,
                                              'VALID', data_format='NWC', name='cnn7'))
            m7 = tf.layers.max_pooling1d(cnn7out, self.poolsize, self.poolstride, padding='SAME')
            alloutput.append(m7)

        self.outputs = tf.concat(alloutput, axis=1)
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

        self.precision, self.recall, self.f1score = calc_performance_metrics(self.ys_, self.prediction)
        tf.summary.scalar('f1', self.f1score)
        self.summary = tf.summary.merge_all()
        return self

