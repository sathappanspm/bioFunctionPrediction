#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
                *.py: Description of what * does.
                Last Modified:
"""

__author__ = "Debanjan Datta"
__email__ = "ddtta@vt.edu"
__version__ = "0.0.1"

import tensorflow as tf
import logging
import pdb

log = logging.getLogger('root.convAE')


# ---------------------------------------------------#

class ConvAutoEncoder(object):

    def __init__(self,
                 vocab_size=24,
                 maxlen=2000,
                 batch_size=128,
                 embedding_dim=256
                 ):

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.set_hyper_parameters()
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.actvn_fn = tf.nn.tanh
        self.emb_matrix = None
        self.set_hyper_parameters()
        self.set_variable_names()

    def build(self):
        self.init_wts()
        self.build_input()
        self.build_encoder_decoder()
        self.build_train()

    # Names for the Tensorflow Graph
    def set_variable_names(self):

        conv_b = []
        conv_w = []
        for i in range( self.num_conv_layers ):
            conv_w.append('conv_w_{}'.format(i))
            conv_b.append('conv_b_{}'.format(i))

        self.name_dict = {
            'emb': ['emb_wt'],
            'conv_w': conv_w,
            'conv_b': conv_b,
        }

    def get_variable_name(self, layer, idx=0):
        return self.name_dict[layer][idx]

    # ------------------------- #
    # place layer parameters here
    # ------------------------- #
    def set_hyper_parameters(self):

        self.num_conv_layers = 5
        self.kernel_size = [
            [3, 2],
            [7, 5],
            [9, 5],
            [11, 7],
            [15, 9]

        ]
        self.num_filters = [32, 16, 16, 8 , 4]
        self.inp_channels = [1, 32, 16, 16, 8 ,4]

        self.strides = [
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [1, 4, 4, 1],
            [1, 4, 2, 1],
            [1, 2, 2, 1],
        ]

    def get_variable(self, shape, name=None):
        with tf.name_scope('weight_or_bias'):
            initial = tf.truncated_normal(
                shape,
                stddev=0.1
            )
            return tf.Variable(initial_value=initial, name=name)

    def init_wts(self):
        # Weights for Embedding Layer
        name = self.get_variable_name('emb')
        self.embed_w = self.get_variable([self.vocab_size + 1, self.embedding_dim], name)

        # Weights and biases for each of the convolutional layer
        self.conv_w = []
        self.conv_b = []
        for i in range(self.num_conv_layers):
            dim = [
                self.kernel_size[i][0],
                self.kernel_size[i][1],
                self.inp_channels[i],
                self.num_filters[i]
            ]
            name = self.get_variable_name('conv_w', i)
            w_i = self.get_variable(dim, name)
            self.conv_w.append(w_i)
            dim = [self.num_filters[i]]
            name = self.get_variable_name('conv_b', i)
            b_i = self.get_variable(dim, name)
            self.conv_b.append(b_i)

    def build_input(self):
        with tf.name_scope('model_input'):
            self.x_input = tf.placeholder(dtype=tf.int64, shape=[None, self.maxlen], name='x')
        return

    def build_encoder_decoder(self):
        with tf.name_scope('Encoder'):

            mask = tf.concat([[0], tf.ones(self.vocab_size)], axis=0)
            self.embmatrix = tf.reshape(mask, shape=[-1, 1]) * self.embed_w
            emb_op = tf.nn.embedding_lookup(self.embmatrix, self.x_input)
            self.emb_op = tf.expand_dims(emb_op, axis=3)

            cur_inp = self.emb_op
            conv_layer_ops = []
            for i in range(self.num_conv_layers):
                _conv_i = tf.nn.conv2d(
                    cur_inp,
                    self.conv_w[i],
                    strides=self.strides[i],
                    padding='SAME'
                ) + self.conv_b[i]
                conv_i = tf.nn.relu(_conv_i)
                # print(conv_i)
                log.info('[Conv AE] Encoder layer i output shape : {}'.format(conv_i.shape))
                conv_layer_ops.append(conv_i)
                print(conv_i.shape)
                cur_inp = conv_i

        self.enc_out = cur_inp
        with tf.name_scope('Decoder'):
            deconv_layer_ops = []
            for i in range(self.num_conv_layers - 1, -1, -1):
                _strides = self.strides[i]
                op_shape = [self.batch_size]
                if i > 0:
                    z = conv_layer_ops[i - 1].get_shape().as_list()[1:4]
                    op_shape.extend(z)
                else:
                    z = [
                        self.maxlen,
                        self.embedding_dim,
                        1
                    ]
                    op_shape.extend(z)

                dec_i = tf.nn.conv2d_transpose(
                    value=conv_layer_ops[i],
                    filter=self.conv_w[i],
                    output_shape=op_shape,
                    strides=_strides,
                    padding="SAME"
                )

                log.info('[Conv AE] Decoder layer i output shape : {}'.format(dec_i.shape))
                print(dec_i.shape)
                deconv_layer_ops.append(dec_i)

        dec_op = deconv_layer_ops[-1]
        cur_op = tf.squeeze(dec_op, axis=-1)

        rev_emb_op = tf.einsum('ijk,kl->ijl', cur_op, tf.transpose(self.embed_w))
        log.info('[Conv AE] Final output shape : ' + str(rev_emb_op.shape))
        self.final_op = rev_emb_op
        return

    def build_train(self):
        _x = self.x_input
        _y = self.final_op
        self.loss1 = tf.losses.sparse_softmax_cross_entropy(labels=_x, logits=_y)
        self.loss2 = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(_x, depth=self.vocab_size + 1), logits=_y)
        ssm = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=_x, logits=_y)
        self.loss = tf.reduce_mean(tf.reduce_mean(ssm, axis=1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train = self.optimizer.minimize(self.loss)
        self.predicted_prob = tf.nn.softmax(_y, axis=-1, name='predicted_seq')
        self.max_out = tf.argmax(self.predicted_prob, axis=-1)

        self.truepos = tf.reduce_sum(tf.cast(self.max_out == _x, dtype=tf.float32))
        self.precision = tf.divide(self.truepos, tf.cast(tf.size(self.max_out), dtype=tf.float32), name='precision')
        return





# m = ConvAutoEncoder(25, 2000)
# m.build()
