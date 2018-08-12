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
import utils
from utils.dataloader import GODAG, FeatureExtractor, load_labelembedding
from utils.dataloader import DataIterator, DataLoader
#from utils.dataloader import vectorized_getlabelmat
from utils import numpy_calc_performance_metrics
from models.encoders import CNNEncoder
from models.rnndecoder import GORNNDecoder
import json
import logging
import os
import numpy as np
from predict import predict_evaluate
import ipdb as pdb
from glob import glob
from tensorflow.python import debug as tf_debug

# handler = logging.FileHandler('{}.log'.format(__processor__))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('root')
# log.addHandler(handler)
FLAGS = tf.app.flags.FLAGS


def create_args():
    tf.app.flags.DEFINE_string(
        'resources',
        './resources',
        "path to data")

    tf.app.flags.DEFINE_string(
        'inputfile',
        '',
        "path to inputfile")

    tf.app.flags.DEFINE_string(
        'outputdir',
        './output',
        "output directory")

    tf.app.flags.DEFINE_string(
        'function',
        '',
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
        32,
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
    tf.app.flags.DEFINE_integer(
        'maxnumfuncs',
        2,
        'maximum number of functions'
    )

    return


def evaluate(predictions, labels):
    labelmat = GODAG.get_fullmat(labels)  # np.any(vectorized_getlabelmat(labels), axis=1)
    predmat = GODAG.get_fullmat(predictions) # np.any(vectorized_getlabelmat(predictions), axis=1)
    return numpy_calc_performance_metrics(labelmat, predmat, threshold=0.2)


def validate(dataiter, sess, encoder, decoder, summary_writer):
    step = 0
    avgPrec, avgRecall, avgF1 = 0.0, 0.0, 0.0
    for x, y in dataiter:
        predictions, summary = sess.run([decoder.predictions, decoder.summary],
                                        feed_dict={decoder.ys_: y[:, :FLAGS.maxnumfuncs],
                                                   encoder.xs_: x,
                                                   decoder.negsamples: np.zeros((y.shape[0], 10))})
        summary_writer.add_summary(summary, step)
        p, r, f1 = evaluate(predictions, y)

        avgPrec += p
        avgRecall += r
        avgF1 += f1
        step += 1

    dataiter.reset()
    return (avgPrec / step, avgRecall / step, avgF1 / step)


def test(dataiter, placeholders, modelpath):
    step = 0
    avgPrec, avgRecall, avgF1 = 0.0, 0.0, 0.0
    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess:
        saver = tf.train.import_meta_graph(glob(os.path.join(modelpath, 'model*meta'))[0])
        saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        log.info('restored model')
        graph = tf.get_default_graph()
        # tf_x, tf_y, tf_thres = graph.get_tensor_by_name('x_input:0'), graph.get_tensor_by_name('y_out:0')
        # tf_thres = graph.get_tensor_by_name('thres:0')
        tf_x, tf_y, tf_neg = [graph.get_tensor_by_name(name) for name in placeholders]
        metrics = [graph.get_tensor_by_name('predictions:0')]
        for x, y in dataiter:
            negsamples = np.zeros((x.shape[0], 10))
            predictions = sess.run(metrics,
                                    feed_dict={tf_y: y[:, :FLAGS.maxnumfuncs],
                                    tf_x: x,
                                    tf_neg: np.zeros((y.shape[0], 10))})
            p, r, f1 = evaluate(predictions, y)
            avgPrec += p
            avgRecall += r
            avgF1 += f1
            step += 1
    return (avgPrec / step, avgRecall / step, avgF1 / step)


def get_negatives(funcs, numNegatives):
    negatives = np.zeros((funcs.shape[0], numNegatives))
    funcmat = GODAG.get_fullmat(funcs)
    for row in range(funcs.shape[0]):
        try:
            # funcmat = GODAG.get_fullmat(funcs[row, :]) # np.any(vectorized_getlabelmat(funcs), axis=1)
            negatives[row, :] = np.random.choice(np.nonzero(~funcmat[row, :])[1], size=numNegatives)
        except:
            pdb.set_trace()

    return negatives


def main(argv):
    _ = GODAG.initialize_idmap(None, None)
    # GO_MAT = GODAG.get_fullmat(goids)
    # log.info('GO Matrix shape - {}'.format(GO_MAT.shape))
    # GO_MAT = np.vstack([np.zeros(GO_MAT.shape[1]), GO_MAT])

    labelembedding, labelIDS = load_labelembedding(os.path.join(FLAGS.resources, 'goEmbeddings.txt'))
    goids = GODAG.initialize_idmap(labelIDS, None)
    # pdb.set_trace()
    assert(labelembedding.shape[0] == len(goids)) , 'label embeddings and known go ids differ'
    labelembeddingsize = labelembedding.shape[1]
    FeatureExtractor.load(FLAGS.resources)
    log.info('Loaded amino acid and ngram mapping data')

    data = DataLoader(filename=FLAGS.inputfile)
    modelsavename = 'savedmodels_{}'.format(int(time.time()))
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        valid_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.validationsize, seqlen=FLAGS.maxseqlen,
                                      dataloader=data, functype=FLAGS.function, featuretype='ngrams',
                                      onlyLeafNodes=True, limit=FLAGS.maxnumfuncs, filterByEvidenceCodes=True,
                                      filename='validation')

        train_iter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.trainsize,
                                  seqlen=FLAGS.maxseqlen, dataloader=data,
                                  functype=FLAGS.function, featuretype='ngrams', onlyLeafNodes=True, limit=FLAGS.maxnumfuncs,
                                  filename='train', filterByEvidenceCodes=True)

        encoder = CNNEncoder(vocab_size=len(FeatureExtractor.ngrammap) + 1, inputsize=train_iter.expectedshape).build()
        log.info('built encoder')
        decoder = GORNNDecoder(encoder.outputs, labelembedding, numfuncs=FLAGS.maxnumfuncs).build()
        log.info('built decoder')
        init = tf.global_variables_initializer()
        init.run(session=sess)
        chkpt = tf.train.Saver(max_to_keep=4)
        train_writer = tf.summary.FileWriter(FLAGS.outputdir + '/train',
                                          sess.graph)

        test_writer = tf.summary.FileWriter(FLAGS.outputdir + '/test')
        step = 0
        maxwait = 5
        wait = 0
        bestf1 = 0
        metagraphFlag = True
        log.info('starting epochs')
        log.info('params - trainsize-{}, validsie-{}, rootfunc-{}, batchsize-{}'.format(FLAGS.trainsize, FLAGS.validationsize,
                                                                                        FLAGS.function, FLAGS.batchsize))
        for epoch in range(FLAGS.num_epochs):
            for x, y in train_iter:
                if x.shape[0] != y.shape[0]:
                    raise Exception('invalid, x-{}, y-{}'.format(str(x.shape), str(y.shape)))

                negatives = get_negatives(y, 10)
                _, loss, summary = sess.run([decoder.train, decoder.loss, decoder.summary],
                                            feed_dict={decoder.ys_: y[:, :FLAGS.maxnumfuncs], encoder.xs_: x,
                                                       decoder.negsamples: negatives})

                train_writer.add_summary(summary, step)
                log.info('step-{}, loss-{}'.format(step, round(loss, 2)))
                step += 1

            log.info('beginning validation')
            prec, recall, f1 = validate(valid_dataiter, sess, encoder, decoder, test_writer)
            log.info('epoch: {} \n precision: {}, recall: {}, f1: {}'.format(epoch,
                                                                             np.round(prec, 2),
                                                                             np.round(recall, 2),
                                                                             np.round(f1, 2)))
            if f1 > (bestf1 + 1e-3):
                bestf1 = f1
                wait = 0
                chkpt.save(sess, os.path.join(FLAGS.outputdir, modelsavename,
                                                'model_{}_{}'.format(FLAGS.function, step)),
                            global_step=step, write_meta_graph=metagraphFlag)
                metagraphFlag = False
            else:
                wait += 1
                if wait > maxwait:
                    log.info('f1 didnt improve for last {} validation steps, so stopping'.format(maxwait))
                    break

            train_iter.reset()

    log.info('testing model')
    test_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.testsize, seqlen=FLAGS.maxseqlen,
                                 dataloader=data, functype=FLAGS.function, featuretype='ngrams',
                                 onlyLeafNodes=True, limit=FLAGS.maxnumfuncs, filename='test',
                                 filterByEvidenceCodes=True)

    placeholders = ['x_in:0', 'y_out:0', 'negsamples:0']
    prec, recall, f1 = test(test_dataiter, placeholders, os.path.join(FLAGS.outputdir, modelsavename))
    log.info('test results')
    log.info('precision: {}, recall: {}, F1: {}'.format(round(prec, 2), round(recall, 2), round(f1, 2)))
    data.close()

if __name__ == "__main__":
    create_args()
    tf.app.run(main)
