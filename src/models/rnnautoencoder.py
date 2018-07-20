#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    RNN autoencoder to identify better representations for input DNA sequences
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


class RNNAutoEncoder(object):
    def __init__(self, vocab_size, embedding_size=256, pretrained_embedding=None, godag=None,
                 lstm_statesize=256, maxlen=2000, finetune_embedding=False, decode_without_input=True):
        self.vocab_size = vocab_size
        self.lstm_statesize = lstm_statesize
        self.embedding_size = embedding_size
        self.godag = godag
        self.pretrained_embedding = pretrained_embedding
        if pretrained_embedding:
            self.vocab_size = self.pretrained_embedding.shape[0]
            self.embedding_size = self.pretrained_embedding.shape[1]

        self.finetune_embedding = finetune_embedding
        self.maxlen = maxlen
        self.decode_without_input = decode_without_input

    def init_variables(self):
        # shape B(atchsize) x L(ength of sequence)
        self.xs_ = tf.placeholder(shape=[None, self.maxlen], dtype=tf.int32, name='x_in')

        # shape vocab_size (number of 3-mers)
        mask = tf.concat([[0], tf.ones(self.vocab_size)], axis=0)

        ## to handle different tensorflow versions
        if hasattr(tf, 'initializers'):
            initializer = tf.initializers.random_uniform
        else:
            initializer = tf.random_uniform_initializer()

        # shape (vocab_size x embedding_size (U))
        if self.pretrained_embedding is not None:
            self.emb = tf.get_variable('emb', initializer=self.pretrained_embedding, dtype=tf.float32,
                                       trainable=self.finetune_embedding)
        else:
            self.emb = tf.get_variable('emb', shape=[self.vocab_size + 1, self.embedding_size])

        self.emb = tf.reshape(mask, shape=[-1, 1]) * self.emb
        self._enc_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_statesize, activation=tf.nn.elu)
        self._dec_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_statesize, activation=tf.nn.elu)

    def build(self):
        self.init_variables()

        #size of the input to the rnn layer is (B x L x U)
        rnn_input = tf.nn.dropout(tf.nn.embedding_lookup(self.emb, self.xs_, name='embed_out'), 0.2, name='embed_dropout')

        # convert rnn input to be in time-major format, so that shape is L x B x U
        rnn_input = tf.transpose(rnn_input, perm=[1, 0, 2])
        # (z_codes, enc_state) = tf.contrib.rnn.static_rnn(self._enc_cell, rnn_input, dtype=tf.float32)
        (z_codes, enc_state) = tf.nn.dynamic_rnn(self._enc_cell, rnn_input, dtype=tf.float32, time_major=True)
        with tf.variable_scope('decoder') as vs:
            dec_weight_ = tf.Variable(tf.truncated_normal([self.lstm_statesize,
                    self.vocab_size], dtype=tf.float32), name='dec_weight'
                    )
            dec_bias_ = tf.Variable(tf.constant(0.1,
                                    shape=[self.vocab_size],
                                    dtype=tf.float32), name='dec_bias')

            if self.decode_without_input:
                dec_inputs = tf.zeros(tf.shape(rnn_input), dtype=tf.float32)
                (dec_outputs, dec_state) = tf.nn.dynamic_rnn(self._dec_cell, dec_inputs, initial_state=enc_state,
                        dtype=tf.float32, time_major=True)

                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
                dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [tf.shape(self.xs_)[0], 1, 1])
                autoencoded_output = (tf.matmul(dec_output_, dec_weight_) + dec_bias_)
            else:
                dec_state = enc_state
                dec_input_ = tf.zeros(tf.shape(rnn_input[0]),
                        dtype=tf.float32)

                dec_outputs = []
                for step in range(rnn_input.shape[0]):
                    if step > 0:
                        vs.reuse_variables()

                    ipdb.set_trace()
                    (dec_input_, dec_state) = \
                        self._dec_cell(dec_input_, dec_state)
                    dec_input_ = tf.matmul(dec_input_, dec_weight_) \
                        + dec_bias_
                    dec_outputs.append(dec_input_)

                autoencoded_output = (tf.transpose(tf.stack(dec_outputs), [1, 0, 2]))

        finalOut = tf.nn.relu(autoencoded_output)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.xs_, logits=finalOut))
        #clippedout = tf.clip_by_value(autoencoded_output, tf.constant(1e-7), 1 - tf.constant(1e-7))
        #ae_logits = tf.log(clippedout / (1 - clippedout))

        #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                       #labels=tf.one_hot(self.xs_, self.vocab_size),
                                       #logits=ae_logits
                                       #)
                                    #)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
        self.train = self.optimizer.minimize(self.loss)
        return self
