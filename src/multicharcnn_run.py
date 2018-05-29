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
__processor__ = 'multiCharCNN'

import time
import pandas as pd
import tensorflow as tf
from utils.dataloader import GODAG, FeatureExtractor
from utils.dataloader import DataIterator, DataLoader, load_pretrained_embedding
from models.multiCharCNN import MultiCharCNN, HierarchicalGODecoder
import json
import logging
import os
import numpy as np
from predict import predict_evaluate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('root')
FLAGS = tf.app.flags.FLAGS
THRESHOLD_RANGE = np.arange(0.1, 0.5, 0.05)


def create_args():
    tf.app.flags.DEFINE_string(
        'resources',
        './resources',
        "path to resources directory")

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
    tf.app.flags.DEFINE_string(
        'inputfile',
        '',
        'inputfile name'
    )
    return


def validate(dataiter, sess, encoder, decoder, summary_writer):
    step = 0
    avgPrec, avgRecall, avgF1 = (np.zeros_like(THRESHOLD_RANGE),
                                 np.zeros_like(THRESHOLD_RANGE),
                                 np.zeros_like(THRESHOLD_RANGE)
                                 )
    for x, y in dataiter:
        prec, recall, f1 = [], [], []
        for thres in THRESHOLD_RANGE:
            p, r, f, summary = sess.run([decoder.precision, decoder.recall,
                                         decoder.f1score, decoder.summary],
                                         feed_dict={decoder.ys_: y, encoder.xs_: x,
                                                    decoder.threshold: [thres]})
            summary_writer.add_summary(summary, step)
            prec.append(p)
            recall.append(r)
            f1.append(f)

        avgPrec += prec
        avgRecall += recall
        avgF1 += f1
        step += 1

    dataiter.reset()
    return (avgPrec / step, avgRecall / step, avgF1 / step)


def main(argv):
    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format(FLAGS.function)))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)

    log.info('GO DAG initialized. Updated function list-{}'.format(len(funcs)))
    FeatureExtractor.load(FLAGS.resources)
    log.info('Loaded amino acid and ngram mapping data')

    data = DataLoader(filename=FLAGS.inputfile)
    modelsavename = 'savedmodels_{}_{}'.format(__processor__, int(time.time()))

    pretrained = None
    featuretype = 'onehot'
    if FLAGS.pretrained != '':
        log.info('loading pretrained embedding')
        pretrained, ngrammap = load_pretrained_embedding(FLAGS.pretrained)
        FeatureExtractor.ngrammap = ngrammap
        featuretype = 'ngrams'

    with tf.Session() as sess:
        valid_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.validationsize,
                                      dataloader=data, functype=FLAGS.function, featuretype=featuretype)


        train_iter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.trainsize,
                                  seqlen=FLAGS.maxseqlen, dataloader=data,
                                  numfiles=np.floor((FLAGS.trainsize * FLAGS.batchsize) / 250000),
                                  functype=FLAGS.function, featuretype=featuretype)

        vocabsize = ((len(FeatureExtractor.ngrammap) + 1) if featuretype == 'ngrams' else
                     (len(FeatureExtractor.aminoacidmap) + 1))

        encoder = MultiCharCNN(vocab_size=vocabsize,
                               inputsize=train_iter.expectedshape,
                               pretrained_embedding=pretrained).build()

        log.info('built encoder')
        decoder = HierarchicalGODecoder(funcs, encoder.outputs, FLAGS.function).build(GODAG)
        log.info('built decoder')
        init = tf.global_variables_initializer()
        init.run(session=sess)
        chkpt = tf.train.Saver(max_to_keep=4)
        train_writer = tf.summary.FileWriter(FLAGS.outputdir + '/train',
                                          sess.graph)

        test_writer = tf.summary.FileWriter(FLAGS.outputdir + '/test')
        step = 0
        maxwait = 1
        wait = 0
        bestf1 = -1
        metagraphFlag = True
        log.info('starting epochs')
        for epoch in range(FLAGS.num_epochs):
            for x, y in train_iter:
                if x.shape[0] != y.shape[0]:
                    raise Exception('invalid, x-{}, y-{}'.format(str(x.shape), str(y.shape)))

                _, loss, summary = sess.run([decoder.train, decoder.loss, decoder.summary],
                                             feed_dict={decoder.ys_: y, encoder.xs_: x,
                                                        decoder.threshold: [0.2]})
                train_writer.add_summary(summary, step)
                log.info('step-{}, loss-{}'.format(step, round(loss, 2)))
                step += 1

            if True:
                log.info('beginning validation')
                prec, recall, f1 = validate(valid_dataiter, sess, encoder, decoder, test_writer)
                thres = np.argmax(np.round(f1, 2))
                log.info('epoch: {} \n precision: {}, recall: {}, f1: {}'.format(epoch,
                                                                                 np.round(prec, 2)[thres],
                                                                                 np.round(recall, 2)[thres],
                                                                                 np.round(f1, 2)[thres]))
                log.info('precision mat {}'.format(str(np.round(prec, 2))))
                log.info('recall mat {}'.format(str(np.round(recall, 2))))
                log.info('f1 mat {}'.format(str(np.round(f1, 2))))

                log.info('selected threshold is {}'.format(thres/10 + 0.1))
                if f1[thres] > (bestf1 + 1e-3):
                    bestf1 = f1[thres]
                    bestthres = THRESHOLD_RANGE[thres]
                    wait = 0
                    chkpt.save(sess, os.path.join(FLAGS.outputdir, modelsavename,
                                                    'model_{}_{}'.format(FLAGS.function, step)),
                                global_step=step, write_meta_graph=metagraphFlag)
                    #metagraphFlag = False
                    metagraphFlag = True
                else:
                    wait += 1
                    if wait > maxwait:
                        log.info('f1 didnt improve for last {} validation steps, so stopping'.format(maxwait))
                        break

                step += 1

            train_iter.reset()

    log.info('testing model')
    test_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.testsize,
                                 dataloader=data, functype=FLAGS.function, featuretype=featuretype)
    prec, recall, f1 = predict_evaluate(test_dataiter, [bestthres], os.path.join(FLAGS.outputdir, modelsavename))
    log.info('test results')
    log.info('precision: {}, recall: {}, F1: {}'.format(round(prec, 2), round(recall, 2), round(f1, 2)))
    data.close()

if __name__ == "__main__":
    create_args()
    tf.app.run(main)
