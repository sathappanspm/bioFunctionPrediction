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
from utils.dataloader import GODAG, FeatureExtractor
from utils.dataloader import DataIterator, DataLoader
# from models.encoders import CNNEncoder
# from models.decoders import HierarchicalGODecoder
import numpy as np
import json
import logging
import os
from glob import glob
import ipdb

# logging.basicConfig(filename='{}.log'.format(__processor__),
                    # filemode='w', level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('root')

FLAGS = tf.app.flags.FLAGS

def create_args():
    tf.app.flags.DEFINE_string(
        'resources',
        './resources',
        "path to data")

    tf.app.flags.DEFINE_string(
        'modelsdir',
        './output/savedmodels/',
        "path to model")

    tf.app.flags.DEFINE_string(
        'function',
        'mf',
        'default function to run'
    )

    tf.app.flags.DEFINE_string(
        'inputfile',
        'testseq.fasta',
        'sequence file'
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
        'testsize',
        100,
        'number of validation batches to use'
    )

    tf.app.flags.DEFINE_boolean(
        'evaluate',
        False,
        'evaluate the results and output precision, recall'
    )
    return


def predict_evaluate(dataiter, thres, modelpath):
    avgPrec, avgRecall, avgF1 = 0.0, 0.0, 0.0
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(glob(os.path.join(modelpath, 'model*meta'))[0])
        saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        log.info('restored model')
        graph = tf.get_default_graph()
        tf_x, tf_y = graph.get_tensor_by_name('x_in:0'), graph.get_tensor_by_name('y_out:0')
        tf_thres = graph.get_tensor_by_name('thres:0')
        metrics = [graph.get_tensor_by_name('precision:0'),
                   graph.get_tensor_by_name('recall:0'),
                   graph.get_tensor_by_name('f1:0')]
        log.info('starting prediction')
        step = 0
        for x, y in dataiter:
            if x.shape[0] != y.shape[0]:
                raise Exception('invalid, x-{}, y-{}'.format(str(x.shape), str(y.shape)))

            prec, recall, f1 = sess.run(metrics, feed_dict={tf_y: y, tf_x: x, tf_thres: thres})
            avgPrec += prec
            avgRecall += recall
            avgF1 += f1
            step += 1

        dataiter.close()
        log.info('read {} test batches'.format(step))
    return avgPrec / step, avgRecall / step, avgF1 / step


def print_predictions(predictions, fastaObjs, gofuncs, thres):
    for row in range(predictions.shape[0]):
        probmat = dict(zip(gofuncs.GOIDS, predictions[row, :]))
        print({'sequenceObj': fastaObjs[row],
               'predictions': {key: val for key, val in probmat.items() if val > thres},
               "model": "DeepGO.v.0.1"})

    return

def predict(dataiter, thres, modelpath, gofuncs):
    log.info('Beginning prediction-{}'.format(modelpath))
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(glob(os.path.join(modelpath, 'model_{}*meta'.format(FLAGS.function)))[0])
        saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        log.info('restored model')
        graph = tf.get_default_graph()
        tf_x, tf_y = graph.get_tensor_by_name('x_in:0'), graph.get_tensor_by_name('y_out:0')
        tf_thres = graph.get_tensor_by_name('thres:0')
        prediction_prob = graph.get_tensor_by_name('prediction:0')
        log.info('starting prediction')
        step = 0
        y = np.zeros((FLAGS.batchsize, len(gofuncs.GOIDS)))
        for x, fastaObjs in dataiter:
            #if x.shape[0] != y.shape[0]:
            #    raise Exception('invalid, x-{}, y-{}'.format(str(x.shape), str(y.shape)))

            predictions = sess.run(prediction_prob, feed_dict={tf_y: y, tf_x: x, tf_thres: [thres]})
            print_predictions(predictions, fastaObjs, gofuncs, thres)
            step += 1

        dataiter.close()
        log.info('read {} test batches'.format(step))
    return



def main(argv):
    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format(FLAGS.function)))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)

    log.info('GO DAG initialized. Updated function list-{}'.format(len(funcs)))
    FeatureExtractor.load(FLAGS.resources)
    log.info('Loaded amino acid and ngram mapping data')

    log.info('Beginning prediction-{}'.format(FLAGS.modelsdir))
    data = DataLoader(filename=FLAGS.inputfile)
    print("read file")
    if FLAGS.evaluate:
        test_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.testsize,
                                     dataloader=data, functype=FLAGS.function, featuretype='ngrams')
        predict_evaluate(test_dataiter, 0.2, FLAGS.modelsdir)
    else:
        test_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.testsize,
                                     dataloader=data, functype=FLAGS.function, featuretype='ngrams', test=True)
        log.info('Beginning prediction')
        predict(test_dataiter, 0.2, FLAGS.modelsdir, GODAG)

if __name__ == "__main__":
    create_args()
    tf.app.run(main)
    # sample run command
    ## python predict.py --modelsdir ./savedmodels/deepGo --resources ./resources --testsize 100 -- batchsize 10 --function bp
