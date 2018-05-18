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
    BIOLOGICAL_PROCESS = 'GO:0008150'
    MOLECULAR_FUNCTION = 'GO:0003674'
    CELLULAR_COMPONENT = 'GO:0005575'
    FUNC_DICT = {'cc': CELLULAR_COMPONENT,
                 'mf': MOLECULAR_FUNCTION,
                 'bp': BIOLOGICAL_PROCESS}

    def __init__(self, inputlayer, labelembedding, num_negatives=10, godag=None,
                 learning_rate=0.001,
                 lstm_statesize=256, numfuncs=5, trainlabelEmbedding=False, distancefunc='cosine'):
        self.inputs = inputlayer
        self.learning_rate = learning_rate
        self.num_neg_samples = num_negatives
        self.label_dimensions = labelembedding.shape[1]
        self.lstm_statesize = lstm_statesize
        self.labelembedding = labelembedding
        self.numfuncs = numfuncs
        self.trainlabelEmbedding = trainlabelEmbedding
        self.godag = godag
        self.distancefunc = distancefunc
        log.info('label embeddings will be trained={}'.format(self.trainlabelEmbedding))
        # self.GO_MAT = GO_MAT

    def init_variables(self):
        # First 5 leaf GO nodes  for a given sequence is only used.
        # size of ys_ is (batchsize x 5)
        self.ys_ = tf.placeholder(shape=[None, self.numfuncs],
                                  dtype=tf.int32, name='y_out')

        self.istraining = tf.placeholder(shape=(1,), dtype=tf.bool, name='trainingFlag')
        # this represents the label embedding, size (GO nodes x labelembeddingsize)
        self.labelemb = tf.get_variable('labelemb', initializer=self.labelembedding, dtype=tf.float32,
                                        trainable=self.trainlabelEmbedding)

        # self.threshold = tf.placeholder(shape=(1,), dtype=tf.float32, name='thres')

        # the negative samples to be used, size (batchsize x number of negatives)
        self.negsamples = tf.placeholder(shape=[None, self.num_neg_samples], dtype=tf.int32, name='negsamples')
        self.lstmcell = tf.contrib.rnn.BasicLSTMCell(self.lstm_statesize, activation=tf.nn.elu)
                                                     # name='lstmcell')

        self.output_weights = tf.get_variable('rnn_outputW', shape=[self.lstm_statesize, self.label_dimensions])
        self.output_bias = tf.get_variable('rnnout_bias', shape=[self.label_dimensions])

        ## to handle different tensorflow versions
        if hasattr(tf, 'initializers'):
            initializer = tf.initializers.random_uniform
        else:
            initializer = tf.random_uniform_initializer()

        self.ytransform = tf.get_variable('ytransform', shape=[self.label_dimensions, self.label_dimensions],
                                          initializer=initializer)

    def build(self):
        self.init_variables()

        ## batchsize x 5 x labelemb
        self.yemb = tf.nn.embedding_lookup(self.labelemb, self.ys_[:, :self.numfuncs], name='yemb')

        ## batchsize x 10 x labelemb
        self.negemb = tf.nn.embedding_lookup(self.labelemb, self.negsamples, name='negemb')
        # rnnin = [tf.zeros(shape=(tf.shape(yemb)[0], 1)) for i in range(5)]
        log.info('input label embedding-{}'.format(self.yemb.get_shape()))
        log.info('negative sample embedding-{}'.format(self.negemb.get_shape()))

        input_norm = tf.layers.batch_normalization(self.inputs, training=self.istraining)
        rnnin = [self.inputs for i in range(self.numfuncs)]
        rnnout, rnn_final_states = tf.nn.static_rnn(self.lstmcell,
                                                    rnnin, dtype=tf.float32)
                                                    #initial_state=self.inputs
                                                    #)
        # log.info('rnnout shape {}'.format(rnnout.get_shape()))
        rflat = tf.reshape(rnnout, shape=[-1, self.lstm_statesize])

        # batchsize*5 x labeldim
        self.output = tf.nn.l2_normalize(tf.nn.softplus(tf.matmul(rflat,
                                                                  self.output_weights)
                                                        + self.output_bias,
                                                        name='yhat'),
                                         dim=1)

        log.info('final decoder out shape {}'.format(self.output.get_shape()))
        ## ipdb.set_trace()
        self.transformed_y = tf.nn.l2_normalize(tf.matmul(tf.reshape(self.yemb, shape=[-1, self.label_dimensions]),
                                                     self.ytransform),
                                           dim=1)

        variable_summaries(self.transformed_y)
        # batch size*10 x labeldim
        self.transformed_negsamples = tf.nn.l2_normalize(tf.matmul(tf.reshape(self.negemb,
                                                                         shape=[-1, self.label_dimensions]),
                                                              self.ytransform),
                                                    dim=1)

        variable_summaries(self.ytransform)

        if self.distancefunc == 'cosine':
            # batchsize *5 x 1
            self.pos_dist = tf.reduce_sum(tf.multiply(self.output, self.transformed_y), axis=1)

            # batchsize *5 x batchsize*10
            self.neg_dist = tf.matmul(self.output, tf.transpose(self.transformed_negsamples))

            # batchsize *5 x 1
        else:
            #ipdb.set_trace()
            self.pos_dist = tf.sqrt(tf.reduce_sum((self.output - self.transformed_y)**2, axis=1))
            self.neg_dist = tf.sqrt(tf.reduce_sum((tf.expand_dims(self.output, axis=1) - self.transformed_negsamples)**2, axis=2))

        log.info('pos_dist-{}, neg_dist-{}'.format(self.pos_dist.get_shape(), self.neg_dist.get_shape()))
        self.min_neg_dist = tf.reduce_min(self.neg_dist, axis=1)
        self.loss  = tf.reduce_mean(tf.exp(self.pos_dist, name='posdist') /
                                    (tf.exp(self.min_neg_dist, name='negdist') + tf.constant(1e-3)),
                                    name='loss')

        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = self.optimizer.minimize(self.loss)

        self.summary = tf.summary.merge_all()
        # self.predictions, self.precision, self.recall, self.f1 = self.make_prediction()
        self.predictions = self.make_prediction()
        return self

    def make_prediction(self):
        # make unit-vectors, size (GO nodes x embeddingsize)
        norm_labelemb = tf.nn.l2_normalize(tf.matmul(self.labelemb, self.ytransform), dim=1, name='labelnorm')

        if self.distancefunc == 'cosine':
            # get cosine similarity, size (batchsize*5 x GO nodes)
            distmat = tf.matmul(self.output, tf.transpose(norm_labelemb), name='pred_dist')
        else:
            distmat = tf.sqrt(tf.reduce_sum((tf.expand_dims(self.output, axis=1) - norm_labelemb)**2, axis=2))

        # boolean matrix of size batchsize x GOlen
        pred_labels = tf.reshape(tf.argmin(distmat, axis=1), shape=[-1, self.numfuncs], name='predictions')

        #truelabels
        # true_labels = GODAG.vfunc(tf.reshape(self.ys_, ))
        # precision, recall, f1 = calc_performance_metrics(pred_labels, true_labels, threshold=0.2)
        return pred_labels



