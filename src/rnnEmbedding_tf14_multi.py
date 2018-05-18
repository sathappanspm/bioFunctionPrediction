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
from utils.dataloader import vectorized_getlabelmat
from utils import numpy_calc_performance_metrics
from models.multiCharCNN import MultiCharCNN
#from models.encoders import CNNEncoder
from models.rnndecoder_tf14 import GORNNDecoder
import json
import logging
import os
import numpy as np
from predict import predict_evaluate
import ipdb
from tensorflow.python import debug as tf_debug

# handler = logging.FileHandler('{}.log'.format(__processor__))
log = logging.getLogger('root')
# log.addHandler(handler)
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
    tf.app.flags.DEFINE_boolean(
        'trainlabel',
        False,
        'flag to switch on label embedding training'
    )
    tf.app.flags.DEFINE_string(
        'distancefunc',
        'cosine',
        'flag to switch on label embedding training'
    )
    return


def evaluate(predictions, labels):
    labelmat = np.any(vectorized_getlabelmat(labels), axis=1)
    predmat = np.any(vectorized_getlabelmat(predictions), axis=1)
    return numpy_calc_performance_metrics(labelmat, predmat, threshold=0.2)


def predict_evaluate(dataiter, modelpath):
    from glob import glob

    avgPrec, avgRecall, avgF1 = 0.0, 0.0, 0.0
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(glob(os.path.join(modelpath, 'model*meta'))[0])
        saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        log.info('restored model')
        graph = tf.get_default_graph()
        tf_x, tf_y = graph.get_tensor_by_name('x_in:0'), graph.get_tensor_by_name('y_out:0')
        tf_neg = graph.get_tensor_by_name('negsamples:0')
        tf_training = graph.get_tensor_by_name('trainingFlag:0')
        metrics = [graph.get_tensor_by_name('predictions:0')]
        log.info('starting prediction')
        step = 0
        for x, y in dataiter:
            if x.shape[0] != y.shape[0]:
                raise Exception('invalid, x-{}, y-{}'.format(str(x.shape), str(y.shape)))

            #ipdb.set_trace()
            predictions = sess.run(metrics, feed_dict={tf_y: y[:, :FLAGS.maxnumfuncs], tf_x: x,
                                                       tf_neg: np.zeros((x.shape[0], 10)),
                                                       tf_training: [False]}
                                    )
            prec, recall, f1 = evaluate(predictions[0], y)
            avgPrec += prec
            avgRecall += recall
            avgF1 += f1
            step += 1

        dataiter.close()
        log.info('read {} test batches'.format(step))
    return avgPrec / step, avgRecall / step, avgF1 / step

def validate(dataiter, sess, encoder, decoder, summary_writer):
    step = 0
    avgPrec, avgRecall, avgF1 = 0.0, 0.0, 0.0
    for x, y in dataiter:
        predictions, summary = sess.run([decoder.predictions, decoder.summary],
                                         feed_dict={decoder.ys_: y[:, :FLAGS.maxnumfuncs], encoder.xs_: x,
                                                decoder.negsamples: np.zeros((y.shape[1], 10)),
                                                decoder.istraining: [False]})

        if summary_writer is not None:
            summary_writer.add_summary(summary, step)

        p, r, f1 = evaluate(predictions, y)

        avgPrec += p
        avgRecall += r
        avgF1 += f1
        step += 1

    dataiter.reset()
    return (avgPrec / step, avgRecall / step, avgF1 / step)


def get_negatives(funcs, numNegatives):
    funcmat = np.any(vectorized_getlabelmat(funcs.astype(int)), axis=1)
    negatives = np.zeros((funcs.shape[0], numNegatives))
    for row in range(funcmat.shape[0]):
        negatives[row, :] = np.random.choice(np.nonzero(~funcmat[row, :])[0], size=numNegatives)

    return negatives


def main(argv):
    goids = GODAG.initialize_idmap(None, None)
    # GO_MAT = GODAG.get_fullmat(goids)
    # log.info('GO Matrix shape - {}'.format(GO_MAT.shape))
    # GO_MAT = np.vstack([np.zeros(GO_MAT.shape[1]), GO_MAT])
    labelembedding = load_labelembedding(os.path.join(FLAGS.data, 'node2vec_emb_go.txt'), goids)
    assert(labelembedding.shape[0] == (len(goids) + 1)) , 'label embeddings and known go ids differ'
    labelembeddingsize = labelembedding.shape[1]
    FeatureExtractor.load(FLAGS.data)
    log.info('Loaded amino acid and ngram mapping data')

    data = DataLoader()
    modelsavename = 'savedmodels_{}'.format(int(time.time()))
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        valid_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.validationsize,
                                      dataloader=data, functype=FLAGS.function, featuretype='onehot',
                                      onlyLeafNodes=True, numfuncs=FLAGS.maxnumfuncs)


        train_iter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.trainsize,
                                  seqlen=FLAGS.maxseqlen, dataloader=data,
                                  numfiles=np.floor((FLAGS.trainsize * FLAGS.batchsize) / 250000),
                                  functype=FLAGS.function, featuretype='onehot', onlyLeafNodes=True, numfuncs=FLAGS.maxnumfuncs)

        #encoder = CNNEncoder(vocab_size=len(FeatureExtractor.ngrammap) + 1, inputsize=train_iter.expectedshape).build()

        encoder = MultiCharCNN(vocab_size=len(FeatureExtractor.aminoacidmap) + 1,
                               inputsize=train_iter.expectedshape, with_dilation=True, charfilter=8,
                               poolsize=80, poolstride=48).build()

        log.info('built encoder')
        decoder = GORNNDecoder(encoder.outputs, labelembedding, numfuncs=FLAGS.maxnumfuncs,
                               trainlabelEmbedding=FLAGS.trainlabel, distancefunc=FLAGS.distancefunc).build()
        log.info('built decoder')

        init = tf.global_variables_initializer()
        init.run(session=sess)
        chkpt = tf.train.Saver(max_to_keep=4)
        train_writer = tf.summary.FileWriter(FLAGS.outputdir + '/train',
                                          sess.graph)

        test_writer = tf.summary.FileWriter(FLAGS.outputdir + '/test')
        step = 0
        maxwait = 2
        wait = 0
        bestf1 = 0
        bestthres = 0
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
                                                decoder.negsamples: negatives, decoder.istraining: [True]})
                train_writer.add_summary(summary, step)
                log.info('step-{}, loss-{}'.format(step, round(loss, 2)))
                step += 1

            log.info('beginning validation')
            prec, recall, f1 = validate(valid_dataiter, sess, encoder, decoder, test_writer)
            log.info('epoch: {} \n precision: {}, recall: {}, f1: {}'.format(epoch,
                                                                             np.round(prec, 2),
                                                                             np.round(recall, 2),
                                                                             np.round(f1, 2)))
            if np.round(f1,2) >= (bestf1):
                bestf1 = np.round(f1,2)
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

            prec, recall, f1 = validate(train_iter, sess, encoder, decoder, None)
            log.info('training error,precision: {}, recall: {}, f1: {}'.format(epoch,
                                                                             np.round(prec, 2),
                                                                             np.round(recall, 2),
                                                                             np.round(f1, 2)))


            train_iter.reset()

    log.info('testing model')
    test_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.testsize,
                                 dataloader=data, functype=FLAGS.function, featuretype='onehot',
                                 onlyLeafNodes=True, numfuncs=FLAGS.maxnumfuncs)
    prec, recall, f1 = predict_evaluate(test_dataiter, os.path.join(FLAGS.outputdir, modelsavename))
    log.info('test results')
    log.info('precision: {}, recall: {}, F1: {}'.format(round(prec, 2), round(recall, 2), round(f1, 2)))
    data.close()

if __name__ == "__main__":
    create_args()
    tf.app.run(main)
