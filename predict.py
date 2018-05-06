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
import json
import logging
import os
from glob import glob

# logging.basicConfig(filename='{}.log'.format(__processor__),
                    # filemode='w', level=logging.DEBUG)

log = logging.getLogger('predict')

FLAGS = tf.app.flags.FLAGS

def create_args():
    tf.app.flags.DEFINE_string(
        'data',
        './data',
        "path to data")

    tf.app.flags.DEFINE_string(
        'modelsdir',
        './output/savedmodels/',
        "path to model")

    tf.app.flags.DEFINE_string(
        'results',
        './results',
        "results directory")

    tf.app.flags.DEFINE_string(
        'function',
        'mf',
        'default function to run'
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


def main(argv):
    funcs = pd.read_pickle(os.path.join(FLAGS.data, '{}.pkl'.format(FLAGS.function)))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)

    log.info('GO DAG initialized. Updated function list-{}'.format(len(funcs)))
    FeatureExtractor.load(FLAGS.data)
    log.info('Loaded amino acid and ngram mapping data')

    data = DataLoader()
    test_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.validationsize,
                                 dataloader=data, functype=FLAGS.function, featuretype='ngrams')

    predict_evaluate(test_dataiter, FLAGS.modelsdir)
    data.close()

if __name__ == "__main__":
    create_args()
    tf.app.run(main)
