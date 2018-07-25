#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
        *.py: Description of what * does.
        Last Modified:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Debanjan Datta"
__version__ = "0.0.1"
__processor__ = 'ConvAE'

import time
import pandas as pd
import tensorflow as tf
from utils.dataloader import GODAG, FeatureExtractor
from utils.dataloader import DataIterator, DataLoader, load_pretrained_embedding
from models.conv_autoencoder_v1_0 import ConvAutoEncoder
import json
import logging
import os
import numpy as np
from predict import predict_evaluate
import ipdb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('root')
FLAGS = tf.app.flags.FLAGS
THRESHOLD_RANGE = np.arange(0.1, 0.5, 0.05)


def create_args( ):
	tf.app.flags.DEFINE_string(
		'resources',
		'./resources',
		"path to resources directory")

	tf.app.flags.DEFINE_string(
		'outputdir',
		'./output',
		"output directory")

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
		'featuretype',
		'onehot',
		'feature to use (onehot or ngrams)'
	)
	tf.app.flags.DEFINE_string(
		'inputfile',
		'',
		'inputfile name'
	)
	return


def validate(dataiter, sess, encoder, decoder, summary_writer):
	step = 0
	avgPrec, avgRecall, avgF1 = (
		np.zeros_like(THRESHOLD_RANGE),
		np.zeros_like(THRESHOLD_RANGE),
		np.zeros_like(THRESHOLD_RANGE)
	)
	for x, y in dataiter:
		prec, recall, f1 = [], [], []
		for thres in THRESHOLD_RANGE:
			p, r, f, summary = sess.run(
				[decoder.precision, decoder.recall, decoder.f1score, decoder.summary],
				feed_dict={
					decoder.ys_: y,
					encoder.xs_: x,
					decoder.threshold: [thres]
				}
			)
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
	funcs = GODAG.initialize_idmap(None, None)

	log.info('GO DAG initialized. Updated function list-{}'.format(len(funcs)))
	FeatureExtractor.load(FLAGS.resources)
	log.info('Loaded amino acid and ngram mapping data')

	data = DataLoader(filename=FLAGS.inputfile)
	modelsavename = 'savedmodels_{}_{}'.format(__processor__, int(time.time()))

	pretrained = None
	featuretype = FLAGS.featuretype

	if FLAGS.pretrained != '':
		log.info('loading pretrained embedding')
		pretrained, ngrammap = load_pretrained_embedding(FLAGS.pretrained)
		FeatureExtractor.ngrammap = ngrammap
		featuretype = 'ngrams'

	log.info('using feature type - {}'.format(featuretype))

	with tf.Session() as sess:
		valid_dataiter = DataIterator(
			batchsize=FLAGS.batchsize,
			size=FLAGS.validationsize,
			dataloader=data,
			functype='',
			featuretype=featuretype
		)

		train_iter = DataIterator(batchsize=FLAGS.batchsize,
								  size=FLAGS.trainsize,
								  seqlen=FLAGS.maxseqlen,
								  dataloader=data,
								  numfiles=np.floor((FLAGS.trainsize * FLAGS.batchsize) / 250000),
								  functype='', featuretype=featuretype)

		vocabsize = ((len(FeatureExtractor.ngrammap)) if featuretype == 'ngrams' else
					 (len(FeatureExtractor.aminoacidmap)))

		cae_model_obj = ConvAutoEncoder(
			vocab_size=vocabsize,
			maxlen=train_iter.expectedshape,
			batch_size=FLAGS.batchsize,
			embedding_dim=256
		)
		cae_model_obj.build()


		init = tf.global_variables_initializer()
		init.run(session=sess)
		chkpt = tf.train.Saver(max_to_keep=4)
		train_writer = tf.summary.FileWriter(
			FLAGS.outputdir + '/train',
			sess.graph
		)

		test_writer = tf.summary.FileWriter(
			FLAGS.outputdir + '/test'
		)
		step = 0
		maxwait = 1
		wait = 0
		metagraphFlag = True
		log.info('starting epochs')

		# -------------------------- #
		#  Start training the autoencoder
		# -------------------------- #
		bestloss = None
		maxwait = 10
		wait = 0
		earlystop = False
		for epoch in range(FLAGS.num_epochs):
			for x, y in train_iter:
				if x.shape[0] != y.shape[0]:
					raise Exception('invalid, x-{}, y-{}'.format(str(x.shape), str(y.shape)))

				_, loss = sess.run(
					[cae_model_obj.train, cae_model_obj.loss],
					feed_dict={
						cae_model_obj.x_input: x
					}
				)

				log.info('step :: {}, loss :: {}'.format(step, round(loss, 3)))
				step += 1
				if step % 1000 == 0:
					x, y = next(valid_dataiter)
					valid_loss = sess.run(
						[cae_model_obj.loss],
						feed_dict={
							cae_model_obj.x_input: x
						}
					)
					log.info('validation loss at step: {} is {}'.format(step, round(valid_loss, 3)))

				if bestloss is None:
					bestloss = loss

				if loss < bestloss:
					wait = 0
					bestloss = loss
				else:
					wait += 1
					if wait > maxwait:
						earlystop = True
						break

			chkpt.save(sess, os.path.join(FLAGS.outputdir, modelsavename,
										  'model_epoch{}'.format(epoch)),
					   global_step=step, write_meta_graph=metagraphFlag)
			train_iter.reset()

			if earlystop:
				log.info('[EARLY STOPPING] Stopping early at epoch: {}, step:{}, loss:{}'.format(epoch, step, bestloss))
				break

	data.close()


if __name__ == "__main__":
	create_args()
	tf.app.run(main)
