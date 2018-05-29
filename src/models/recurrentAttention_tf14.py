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
from .customRNN import LocationDecoder

log = logging.getLogger('root.RNNDecoder')


class GORNNDecoder(object):
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
        #self.lstmcell = tf.contrib.rnn.BasicLSTMCell(self.lstm_statesize, activation=tf.nn.elu)

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

        input_norm = tf.layers.batch_normalization(self.inputs, training=self.istraining)
        self.locpredictor = LocationDecoder(self.inputs, self.yemb,
                                            init_loc=tf.nn.embedding_lookup(self.labelemb, self.godag.get_id('GO:0008150')))
        ## batchsize x 10 x labelemb
        self.negemb = tf.nn.embedding_lookup(self.labelemb, self.negsamples, name='negemb')
        log.info('input label embedding-{}'.format(self.yemb.get_shape()))
        log.info('negative sample embedding-{}'.format(self.negemb.get_shape()))

        decoderloc, decoderhidden = self.locpredictor.unroll()
        log.info('decoderloc-{}, decoderhidden-{}'.format(decoderloc.get_shape(), decoderhidden.get_shape()))

        log.info('before flat-{}'.format(decoderloc.get_shape()))
        rflat = tf.reshape(decoderloc, shape=[-1, self.label_dimensions + 1])
        log.info('after flat-{}'.format(rflat.get_shape()))

        predlabels = rflat[:, :-1]
        self.actionpreds = tf.nn.sigmoid(rflat[:, -1], name='action')

        log.info('actions-{}, predictions-{}'.format(predlabels.get_shape(),  self.actionpreds.get_shape()))
        #ipdb.set_trace()
        actionmask = tf.reshape(tf.cast(tf.logical_not(tf.equal(self.ys_, 0)), dtype=tf.float32), shape=[-1, 1])

        actionerr = tf.reduce_mean((actionmask - self.actionpreds)**2)

        # batchsize*5 x labeldim
        self.output = tf.nn.l2_normalize(predlabels, dim=1)
        log.info('final decoder out shape {}'.format(self.output.get_shape()))
        ## ipdb.set_trace()
        #self.transformed_y = tf.nn.l2_normalize(tf.matmul(tf.reshape(self.yemb, shape=[-1, self.label_dimensions]),
                                                          #self.ytransform), dim=1)
        #ipdb.set_trace()
        self.transformed_y = tf.nn.l2_normalize(tf.reshape(self.yemb,
                                                           shape=[-1, self.label_dimensions]),
                                                dim=1)

        #variable_summaries(self.transformed_y)
        # batch size*10 x labeldim
        self.transformed_negsamples = tf.nn.l2_normalize(tf.matmul(tf.reshape(self.negemb,
                                                                         shape=[-1, self.label_dimensions]),
                                                              self.ytransform),
                                                    dim=1)

        #self.transformed_negsamples = tf.nn.l2_normalize(tf.reshape(self.negemb,
        #                                                            shape=[-1, self.label_dimensions]),
        #                                                 dim=1)
        #variable_summaries(self.ytransform)
        log.info('ty-{}, tneg-{}'.format(self.transformed_y.get_shape(), self.transformed_negsamples.get_shape()))
        if self.distancefunc == 'cosine':
            # batchsize *5 x 1
            self.pos_dist = tf.abs(tf.reduce_sum(tf.multiply(self.output, self.transformed_y), axis=1))
            self.labelembloss = -1 * tf.reduce_mean(tf.multiply(actionmask, self.pos_dist), name='loss')

            # batchsize *5 x batchsize*10
            #self.neg_dist = tf.abs(tf.matmul(self.output, tf.transpose(self.transformed_negsamples)))

            # batchsize *5 x 1
        else:
            #ipdb.set_trace()
            log.info('using euclidean distance')
            self.pos_dist = tf.sqrt(tf.reduce_sum((self.output - self.transformed_y)**2, axis=1))
            self.neg_dist = tf.sqrt(tf.reduce_sum((tf.expand_dims(self.output, axis=1) - self.transformed_negsamples)**2, axis=2))

            log.info('pos_dist-{}, neg_dist-{}'.format(self.pos_dist.get_shape(), self.neg_dist.get_shape()))
            self.min_neg_dist = tf.reduce_mean(self.neg_dist, axis=1)

            #ipdb.set_trace()
            self.labelembloss  = tf.exp(self.pos_dist, name='posdist') / (tf.exp(self.min_neg_dist, name='negdist') + tf.constant(1e-7))
            self.labelembloss = tf.reduce_mean(tf.multiply(actionmask, self.labelembloss), name='loss')

        self.loss = self.labelembloss + actionerr

        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = self.optimizer.minimize(self.loss)

        self.summary = tf.summary.merge_all()
        # self.predictions, self.precision, self.recall, self.f1 = self.make_prediction()
        self.predictions = self.make_prediction()
        log.info('prediction shape-{}'.format(self.predictions.get_shape()))
        return self

    def make_prediction(self):
        # make unit-vectors, size (GO nodes x embeddingsize)
        norm_labelemb = tf.nn.l2_normalize(tf.matmul(self.labelemb, self.ytransform), dim=1, name='labelnorm')

        if self.distancefunc == 'cosine':
            # get cosine similarity, size (batchsize*5 x GO nodes)
            distmat = tf.abs(tf.matmul(self.output, tf.transpose(norm_labelemb)), name='pred_dist')
            pred_labels = tf.reshape(tf.argmax(distmat, axis=1), shape=[-1, self.numfuncs], name='predictions')
        else:
            distmat = tf.sqrt(tf.reduce_sum((tf.expand_dims(self.output, axis=1) - norm_labelemb)**2, axis=2), name='pred_dist')
            # boolean matrix of size batchsize x GOlen
            pred_labels = tf.reshape(tf.argmin(distmat, axis=1), shape=[-1, self.numfuncs], name='predictions')

        #truelabels
        # true_labels = GODAG.vfunc(tf.reshape(self.ys_, ))
        # precision, recall, f1 = calc_performance_metrics(pred_labels, true_labels, threshold=0.2)
        return pred_labels



