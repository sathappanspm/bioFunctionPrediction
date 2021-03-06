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
from models.recurrentAttention_tf14 import GORNNDecoder
import json
import logging
import os
import numpy as np
from predict import predict_evaluate
import ipdb
from tensorflow.python import debug as tf_debug

# handler = logging.FileHandler('{}.log'.format(__processor__))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('root')
# log.addHandler(handler)
FLAGS = tf.app.flags.FLAGS


def create_args():
    tf.app.flags.DEFINE_string(
        'resources',
        './data',
        "path to data")

    tf.app.flags.DEFINE_string(
        'inputfile',
        './data',
        "path to sequence file")

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
        'predict',
        "",
        'model path to indicate no training'
    )

    tf.app.flags.DEFINE_string(
        'distancefunc',
        'cosine',
        'flag to switch on label embedding training'
    )
    return


def evaluate(predictions, labels, action=None):
    #ipdb.set_trace()
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
        metrics = [graph.get_tensor_by_name('predictions:0'), graph.get_tensor_by_name('pred_dist:0')]
                   #graph.get_tensor_by_name('action:0')]
        log.info('starting prediction')
        step = 0
        for x, y in dataiter:
            if x.shape[0] != y.shape[0]:
                raise Exception('invalid, x-{}, y-{}'.format(str(x.shape), str(y.shape)))

            predictions = sess.run(metrics, feed_dict={tf_y: y[:, :FLAGS.maxnumfuncs], tf_x: x,
                                                       tf_neg: np.zeros((x.shape[0], 10)),
                                                       tf_training: [False]}
                                    )
            ipdb.set_trace()
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
        predictions, actions, summary = sess.run([decoder.predictions, decoder.actionpreds, decoder.summary],
                                         feed_dict={decoder.ys_: y[:, :FLAGS.maxnumfuncs], encoder.xs_: x,
                                                decoder.negsamples: np.zeros((y.shape[1], 10)),
                                                decoder.istraining: [False]})

        if summary_writer is not None:
            summary_writer.add_summary(summary, step)

        p, r, f1 = evaluate(predictions, y, action=actions)

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

    labelembedding = load_labelembedding(os.path.join(FLAGS.resources, 'goEmbeddings.txt'), goids)
    assert(labelembedding.shape[0] == (len(goids))) , 'label embeddings and known go ids differ'

    ## Add a row of zeros to refer to NOGO or STOPGO
    labelembedding = np.vstack([np.zeros(labelembedding.shape[1]), labelembedding]).astype(np.float32)
    labelembeddingsize = labelembedding.shape[1]

    # shift all goids by 1, to allow STOPGO
    GODAG.idmap = {key: (val + 1) for key, val in GODAG.idmap.items()}
    log.info('min go index - {}'.format(min(list(GODAG.idmap.values()))))
    GODAG.idmap['STOPGO'] = 0
    GODAG.GOIDS.insert(0, 'STOPGO')
    log.info('first from main-{}, from goids-{},  from idmap-{}, by reversemap-{}'.format(goids[0], GODAG.GOIDS[1], GODAG.id2node(1), GODAG.get_id(goids[0])))

    FeatureExtractor.load(FLAGS.resources)
    log.info('Loaded amino acid and ngram mapping data')

    data = DataLoader(filename=FLAGS.inputfile)
    modelsavename = FLAGS.predict
    if FLAGS.predict == "":
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
                                   inputsize=train_iter.expectedshape, with_dilation=False, charfilter=32,
                                   poolsize=80, poolstride=48).build()

            log.info('built encoder')
            decoder = GORNNDecoder(encoder.outputs, labelembedding, numfuncs=FLAGS.maxnumfuncs,
                                   trainlabelEmbedding=FLAGS.trainlabel, distancefunc=FLAGS.distancefunc, godag=GODAG).build()
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
                    log.info('saving meta graph')
                    #ipdb.set_trace()
                    chkpt.save(sess, os.path.join(FLAGS.outputdir, modelsavename,
                                                    'model_{}_{}'.format(FLAGS.function, step)),
                                global_step=step, write_meta_graph=metagraphFlag)
                    metagraphFlag = True
                else:
                    wait += 1
                    if wait > maxwait:
                        log.info('f1 didnt improve for last {} validation steps, so stopping'.format(maxwait))
                        break

                train_iter.reset()
                prec, recall, f1 = validate(train_iter, sess, encoder, decoder, None)
                log.info('training error,epoch-{}, precision: {}, recall: {}, f1: {}'.format(epoch,
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
