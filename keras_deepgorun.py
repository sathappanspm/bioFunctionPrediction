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

import time
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from utils.dataloader import GODAG, FeatureExtractor
from utils.dataloader import DataIterator, DataLoader
from models.deepgo import KerasDeepGO
import json
import logging
import os

# log = logging.basicConfig(filename='{}.log'.format(__processor__),
                    # level=logging.DEBUG)

log = logging.getLogger('main')

logging.basicConfig(filename='deepgo.log', format='%(levelname)s:%(message)s', level=logging.INFO)
FLAGS = tf.app.flags.FLAGS

def create_args():
    tf.app.flags.DEFINE_string(
        'data',
        './data',
        "path to data")

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


def main(argv):
    funcs = pd.read_pickle(os.path.join(FLAGS.data, '{}.pkl'.format(FLAGS.function)))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)

    log.info('GO DAG initialized. Updated function list-{}'.format(len(funcs)))
    FeatureExtractor.load(FLAGS.data)
    log.info('Loaded amino acid and ngram mapping data')

    with tf.Session() as sess:
        data = DataLoader()
        valid_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=10112/FLAGS.batchsize,
                                      dataloader=data, functype=FLAGS.function, featuretype='ngrams', all_labels=False)

        train_iter = DataIterator(batchsize=FLAGS.batchsize, size=250112/FLAGS.batchsize,
                                  seqlen=FLAGS.maxseqlen, dataloader=data, numfiles=4, numfuncs=len(funcs),
                                  functype=FLAGS.function, featuretype='ngrams', all_labels=False,)


        model = KerasDeepGO(funcs, FLAGS.function, GODAG, train_iter.expectedshape, len(FeatureExtractor.ngrammap)).build()
        log.info('built encoder')
        log.info('built decoder')
        keras.backend.set_session(sess)
        log.info('starting epochs')

        model_path = FLAGS.outputdir + 'models/model_seq_' + FLAGS.function + '.h5'
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            verbose=1, save_best_only=True)
        earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

        model.fit_generator(
            train_iter,
            steps_per_epoch=int(250112/128),
            epochs=5,
            validation_data=valid_dataiter,
            validation_steps=int(10112/128),
            max_queue_size=128,
            callbacks=[checkpointer, earlystopper])

        valid_dataiter.close()
        train_iter.close()
        data.close()

if __name__ == "__main__":
    create_args()
    tf.app.run(main)
