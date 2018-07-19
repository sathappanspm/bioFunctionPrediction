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
import logging
import ipdb
log = logging.getLogger('root.convAE')


class ConvAutoEncoder(object):
    def __init__(self, embedding_size=64, vocab_size=24,
                 maxlen=2000):
        self.embedding_dim = embedding_size
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.actvn_fn = tf.nn.tanh

    def build_encoder(self):
        # input sequence of amino acids (mostly maxlen of 2000 is used), shape = batchsize x maxlen
        self.xs_ = tf.placeholder(shape=[None, self.maxlen], dtype=tf.int32, name='xs_')

        # create feature embedding (this is done at character level or ngram level based on kind of input)
        # shape batchsize x maxlen x embedding_size
        #mask = tf.concat([[0], tf.ones(self.vocab_size - 1)], axis=0)
        with tf.name_scope('aminoAcid_emb_layer'):
            self.embmatrix = tf.get_variable('emb', [self.vocab_size, self.embedding_dim],
                                       dtype=tf.float32)
            #self.embmatrix = tf.reshape(mask, shape=[-1, 1]) * self.embmatrix
            self.emblayer = tf.nn.embedding_lookup(self.embmatrix, self.xs_)


        # assuming embedding dimension is 64, and maxlen in 2000
        # shape is batch x 1998 x 62 x 32
        with tf.name_scope('enc_conv1'):
            self.conv1 = tf.layers.conv2d(tf.expand_dims(self.emblayer, 3),
                                          filters=32, kernel_size=(3,3), strides=(1,1),
                                          activation=self.actvn_fn, use_bias=True, name='conv1')

        # batch x 666 x 62 x 16
        with tf.name_scope('enc_conv2'):
            self.conv2 = tf.layers.conv2d(self.conv1, 16, (5, 5), strides=(3,1), activation=self.actvn_fn,
                                          padding='SAME', name='conv2')

        # batch x 167 x 31 x 8
        with tf.name_scope('enc_conv3'):
            self.conv3 = tf.layers.conv2d(self.conv2, 8, (7, 3), strides=(4, 2), activation=self.actvn_fn,
                                          padding='SAME', name='conv3')

        # batch x 34 x 11 x 4
        with tf.name_scope('enc_conv4'):
            self.conv4 = tf.layers.conv2d(self.conv3, 4, (9, 5), strides=(5, 3), activation=self.actvn_fn,
                                          padding='SAME', name='conv4')



        self.sequence_embedding = tf.layers.dense(tf.contrib.layers.flatten(self.conv4), 1024)
        return self.sequence_embedding

    def build_decoder(self, encoderout):
        dec_input = tf.reshape(tf.layers.dense(encoderout, 1496), shape=[-1, 34, 11, 4])

        # Shape is batch x 170 x 33 x 8
        with tf.name_scope('decoder_conv1'):
            self.dec_conv1 = tf.layers.conv2d_transpose(dec_input, 8, (9, 5), strides=(5, 3),
                                      activation=self.actvn_fn, padding='SAME', name='dec_conv1')

        # shape is batch x 680 x 66 x 16
        with tf.name_scope('decoder_conv2'):
            self.dec_conv2 = tf.layers.conv2d_transpose(self.dec_conv1, 16, (7, 3), strides=(4, 2),
                                      activation=self.actvn_fn, padding='SAME', name='dec_conv2')

        # shape is batch x 2040 x 66 x 32
        with tf.name_scope('decoder_conv3'):
            self.dec_conv3 = tf.layers.conv2d_transpose(self.dec_conv2, 32, (5, 5), strides=(3, 1),
                                      activation=self.actvn_fn, padding='SAME', name='dec_conv3')


        # shape is batch x 2000 x vocab_size
        self.dec_output = tf.squeeze(tf.layers.conv2d(self.dec_conv3, 1, (41, 43), strides=(1,1), activation=self.actvn_fn))

        return self.dec_output

    def build(self):
        encoder = self.build_encoder()
        decoder = self.build_decoder(encoder)
        #clippedout = tf.clip_by_value(decoder, tf.constant(1e-7), 1 - tf.constant(1e-7))
        #logits = tf.log(clippedout / (1 - clippedout))

        #lbls = tf.one_hot(self.xs_, depth=self.vocab_size, dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.xs_, logits=decoder))
        #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=lbls[:, :, 1:], logits=logits))

        tf.summary.scalar('loss', self.loss)
        #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
        self.train = self.optimizer.minimize(self.loss)
        self.summary = tf.summary.merge_all()
        return self


