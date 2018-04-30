#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"
__processor__ = 'deepGO'

import pandas as pd
import tensorflow as tf
from utils.dataloader import GODAG, FeatureExtractor
from utils.dataloader import DataIterator, DataLoader
from models.encoders import CNNEncoder
from models.decoders import HierarchicalGODecoder
import json
import logging
import os

logging.basicConfig(filename='{}.log'.format(__processor__),
                    level=logging.DEBUG)

log = logging.getLogger('main')

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
    funcs = GODAG.initialize_idmap(funcs)
    
    log.info('GO DAG initialized. Updated function list-{}'.format(len(funcs)))
    FeatureExtractor.load(FLAGS.data)
    log.info('Loaded amino acid and ngram mapping data')
    
    with tf.Session() as sess:
        data = DataLoader()
        valid_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.trainsize,
                                      dataloader=data, functype=FLAGS.function, featuretype='ngrams')

        train_iter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.validationsize,
                                  seqlen=FLAGS.maxseqlen, dataloader=data, numfiles=4,
                                  functype=FLAGS.function, featuretype='ngrams')

        encoder = CNNEncoder(vocab_size=len(FeatureExtractor.ngrammap), inputsize=train_iter.expectedshape).build()
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
        maxwait = 4
        wait = 0
        bestf1 = 0
        metagraphFlag = True
        log.info('starting epochs')
        for epoch in range(FLAGS.num_epochs):
            for x, y in train_iter:
                log.info('data-{}, labels-{}'.format(x.shape, y.shape))
                _, loss, summary = sess.run([decoder.train, decoder.loss, decoder.summary],
                                            feed_dict={decoder.ys_: y, encoder.xs_: x})
                train_writer.add_summary(summary, step)

                if step % 100 == 0:
                    prec, recall, f1, summary = sess.run([decoder.precision, decoder.recall,
                                                  decoder.f1score, decoder.summary],
                                                 feed_dict={decoder.ys_: y, encoder.xs_: x})
                    test_writer.add_summary(summary, step)
                    f1 = round(f1, 2)
                    log.info('epoch: {} \n precision: {}, recall: {}, f1: {}'.format(epoch,
                                                                                     round(precision, 2),
                                                                                     round(recall, 2), f1))
                    if f1 > bestf1:
                        bestf1 = f1
                        wait = 0
                        chkpt.save(sess, os.path.join(OUTDIR, 'savedmodels',
                                                      'model_{}'.format(start_date)),
                                   global_step=step, write_meta_graph=metagraphFlag)
                        metagraphFlag = False

                    else:
                        wait += 1
                        if wait > maxwait:
                            log.info('f1 didnt improve for last {} validation steps, so stopping')
                            break

            valid_dataiter.reset()
            train_iter.reset()

        data.close()

if __name__ == "__main__":
    create_args()
    tf.app.run(main)
