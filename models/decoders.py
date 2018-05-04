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

log = logging.getLogger('Decoder')

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
        var = tf.get_variable(name, shape=[self.inputs.shape[-1], 1], dtype=tf.float32)
        bias = tf.get_variable('{}_b'.format(name), shape=[1], dtype=tf.float32)
        return tf.sigmoid(tf.matmul(self.inputs, var) + bias, name='{}_out'.format(name))

    def init_variables(self, godag):
        self.ys_ = tf.placeholder(shape=[None, len(godag.GOIDS)],
                                  dtype=tf.float32, name='y_out')
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
