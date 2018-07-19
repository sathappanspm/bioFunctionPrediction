#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"
__processor__ = 'deepGO'

import logging
import time
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import keras
from tensorflow.contrib.keras import backend as K
from utils.dataloader import GODAG, FeatureExtractor
from utils.dataloader import DataIterator, DataLoader
import json
import os
from utils import numpy_calc_performance_metrics
import numpy as np
import sys
import ipdb

logging.basicConfig(level=logging.INFO)
strhandler = logging.StreamHandler(sys.stdout)
log = logging.getLogger('root')
log.addHandler(strhandler)
#logging.basicConfig(filename='deepgo.log', format='%(levelname)s:%(message)s', level=logging.INFO)
FLAGS = tf.app.flags.FLAGS
original_dim = 0

def create_args():
    tf.app.flags.DEFINE_string(
        'resources',
        './resources',
        "path to resources")

    tf.app.flags.DEFINE_string(
        'inputfile',
        './data',
        "path to sequence input")

    tf.app.flags.DEFINE_string(
        'outputdir',
        './output',
        "output directory")

    tf.app.flags.DEFINE_string(
        'function',
        'mf',
        'default function to run'
    )
    tf.app.flags.DEFINE_integer(
        'trainsize',
        2000,
        'number of train batches'
    )
    tf.app.flags.DEFINE_integer(
        'batchsize',
        128,
        'size of batch'
    )
    tf.app.flags.DEFINE_integer(
        'maxseqlen',
        2000,
        'maximum sequence length'
    )

    tf.app.flags.DEFINE_integer(
        'validationsize',
        100,
        'number of validation batches to use'
    )

    tf.app.flags.DEFINE_integer(
        'num_epochs',
        5,
        'number of epochs'
    )
    return


epsilon_std = 1.0
np.random.seed(42)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random_normal(shape=(len(args),), mean=0.,
                                      stddev=epsilon_std)
    return z_mean + tf.exp(z_log_var / 2) * epsilon


def get_AE(original_dim, max_features, maxlen):
    latent_dim = 2
    intermediate_dim = 1024
    x = keras.layers.Input(shape=(original_dim,), dtype='int32', name='i1')
    emb = keras.layers.Embedding(max_features, 128, input_length=maxlen)(x)
    ipdb.set_trace()
    h = keras.layers.Dense(intermediate_dim,activation="elu")(emb)
    h= keras.layers.Dropout(0.7)(h)
    h = keras.layers.Dense(intermediate_dim, activation='elu')(h)
    h= keras.layers.BatchNormalization(scale=True, axis=1)(h)
    h = keras.layers.Dense(intermediate_dim, activation='elu')(h)

    #Latent layers
    z_mean=keras.layers.Dense(latent_dim)(h)
    z_log_var=keras.layers.Dense(latent_dim)(h)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    #Decoding layers

    decoder_1= keras.layers.Dense(intermediate_dim, activation='elu')
    decoder_2=keras.layers.Dense(intermediate_dim, activation='elu')
    decoder_2d=keras.layers.Dropout(0.7)
    decoder_3=keras.layers.Dense(intermediate_dim, activation='elu')
    decoder_out=keras.layers.Dense(original_dim, activation='sigmoid')
    x_decoded_mean = decoder_out(decoder_3(decoder_2d(decoder_2(decoder_1(z)))))

    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * keras.losses.categorical_crossentropy(x,  x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae = keras.models.Model(x, x_decoded_mean)
    vae.compile(optimizer="adam", loss=vae_loss,metrics=["categorical_accuracy"])
    return vae


def datawrapper(dataiter):
    for x, y in dataiter:
        yield x, x


def main(argv):
    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format(FLAGS.function)))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)

    log.info('GO DAG initialized. Updated function list-{}'.format(len(funcs)))
    FeatureExtractor.load(FLAGS.resources)
    log.info('Loaded amino acid and ngram mapping data')
    with tf.Session() as sess:
        data = DataLoader(filename=FLAGS.inputfile)
        log.info('initializing validation data')
        valid_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.validationsize,
                                      dataloader=data, functype=FLAGS.function, featuretype='onehot',numfuncs=len(funcs),
                                      all_labels=False, autoreset=True)

        log.info('initializing train data')
        train_iter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.trainsize,
                                  seqlen=FLAGS.maxseqlen, dataloader=data, numfiles=4, numfuncs=len(funcs),
                                  functype=FLAGS.function, featuretype='onehot', all_labels=False, autoreset=True)

        global original_dim
        original_dim = train_iter.expectedshape

        model = get_AE(train_iter.expectedshape, len(FeatureExtractor.aminoacidmap), train_iter.expectedshape)
        keras.backend.set_session(sess)
        log.info('starting epochs')

        model_path = FLAGS.outputdir + 'models/model_seq_' + FLAGS.function + '.h5'
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            verbose=1, save_best_only=True, save_weights_only=True)
        earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

        model_jsonpath = FLAGS.outputdir + 'models/model_{}.json'.format(FLAGS.function)
        f = open(model_jsonpath, 'w')
        f.write(model.to_json())
        f.close()

        model.fit_generator(
            datawrapper(train_iter),
            steps_per_epoch=FLAGS.trainsize,
            epochs=5,
            validation_data=datawrapper(valid_dataiter),
            validation_steps=FLAGS.validationsize,
            callbacks=[checkpointer, earlystopper])

        valid_dataiter.close()
        train_iter.close()
        data.close()


if __name__ == "__main__":
    create_args()
    tf.app.run(main)
