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
try:
    from tensorflow import keras
except:
    import keras
from utils.dataloader import GODAG, FeatureExtractor
from utils.dataloader import DataIterator, DataLoader, load_pretrained_embedding
from models.deepgo import KerasDeepGO
import json
import os
from utils import numpy_calc_performance_metrics
import numpy as np
import sys

logging.basicConfig(level=logging.INFO)
strhandler = logging.StreamHandler(sys.stdout)
log = logging.getLogger('root')
log.addHandler(strhandler)
#logging.basicConfig(filename='deepgo.log', format='%(levelname)s:%(message)s', level=logging.INFO)
FLAGS = tf.app.flags.FLAGS
THRESHOLD_RANGE = np.arange(0.1, 0.5, 0.05)


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
        'testsize',
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

    tf.app.flags.DEFINE_string(
        'pretrained',
        '',
        'location of pretrained embedding'
    )
    return


def predict_evaluate(test_dataiter, model_jsonpath, modelpath):
    avgPrec, avgRecall, avgF1 = (np.zeros_like(THRESHOLD_RANGE),
                                 np.zeros_like(THRESHOLD_RANGE),
                                 np.zeros_like(THRESHOLD_RANGE)
                                 )
    step = 0
    with tf.Session() as sess:
        with open(model_jsonpath) as inf:
            model = keras.models.model_from_json(inf.read())
        model.load_weights(modelpath)
        for x, y in test_dataiter:
            prec, recall, f1 = [], [], []
            for thres in THRESHOLD_RANGE:
                preds = model.predict_on_batch(x)
                preds = np.concatenate([preds, np.zeros((preds.shape[0], y.shape[1] - preds.shape[1]))], axis=1)
                p, r, f = numpy_calc_performance_metrics(y, preds, thres)
                prec.append(p)
                recall.append(r)
                f1.append(f)

            step += 1
            avgPrec += prec
            avgRecall += recall
            avgF1 += f1

    return (avgPrec / step, avgRecall / step, avgF1 / step)


def main(argv):
    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format(FLAGS.function)))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)

    log.info('GO DAG initialized. Updated function list-{}'.format(len(funcs)))
    FeatureExtractor.load(FLAGS.resources)
    log.info('Loaded amino acid and ngram mapping data')
    pretrained = None
    featuretype = 'onehot'
    if FLAGS.pretrained != '':
        log.info('loading pretrained embedding')
        pretrained, ngrammap = load_pretrained_embedding(FLAGS.pretrained)
        FeatureExtractor.ngrammap = ngrammap
        featuretype = 'ngrams'

    with tf.Session() as sess:
        data = DataLoader(filename=FLAGS.inputfile)
        log.info('initializing validation data')
        valid_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.validationsize,
                                      dataloader=data, functype=FLAGS.function, featuretype='ngrams',numfuncs=len(funcs),
                                      all_labels=False, filename='validation', filterByEvidenceCodes=True)

        log.info('initializing train data')
        train_iter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.trainsize,
                                  seqlen=FLAGS.maxseqlen, dataloader=data, numfiles=4,
                                  numfuncs=len(funcs), functype=FLAGS.function,
                                  featuretype='ngrams', all_labels=False, autoreset=True,
                                  filename='train')

        vocabsize = ((len(FeatureExtractor.ngrammap) + 1) if featuretype == 'ngrams' else
                     (len(FeatureExtractor.aminoacidmap) + 1))

        model = KerasDeepGO(funcs, FLAGS.function, GODAG, train_iter.expectedshape, vocabsize, embedding_size=256, pretrained_embedding=pretrained).build()
        log.info('built encoder')
        log.info('built decoder')
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
            train_iter,
            steps_per_epoch=FLAGS.trainsize,
            epochs=5,
            validation_data=valid_dataiter,
            validation_steps=FLAGS.validationsize,
            max_queue_size=128,
            callbacks=[checkpointer, earlystopper])

        valid_dataiter.close()
        train_iter.close()

    log.info('initializing test data')
    test_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.testsize,
                                 seqlen=FLAGS.maxseqlen, dataloader=data, numfiles=4, numfuncs=len(funcs),
                                 functype=FLAGS.function, featuretype='ngrams', filename='test', all_labels=True)

    prec, recall, f1 = predict_evaluate(test_dataiter, model_jsonpath, model_path)
    log.info('testing error, prec-{}, recall-{}, f1-{}'.format(np.round(prec, 3), np.round(recall, 3), np.round(f1, 3)))
    data.close()

if __name__ == "__main__":
    create_args()
    tf.app.run(main)
