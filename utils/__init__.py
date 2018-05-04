#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"

import tensorflow as tf
from collections import deque
import logging


def calc_performance_metrics(labels, predictions, threshold=0.35):
    labels = tf.cast(labels, tf.bool)
    predictions = predictions > threshold
    tp = tf.reduce_sum(tf.cast(tf.logical_and(labels, predictions), tf.float32), axis=1)
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(labels), predictions), tf.float32), axis=1)
    fn = tf.reduce_sum(tf.cast(tf.logical_and(labels, tf.logical_not(predictions)), tf.float32), axis=1)
    precision = tp / (tp + fp + tf.constant(1e-7))
    recall = tp / (tp + fn + tf.constant(1e-7))
    f1 = 2 * (precision * recall) / (precision + recall + tf.constant(1e-7))
    return tf.reduce_mean(precision), tf.reduce_mean(recall), tf.reduce_mean(f1)
    # tf.logging.info('precision- {}'.format(tf.reduce_sum(precision)))
    # return tf.reduce_sum(tp), tf.reduce_sum(fp), tf.reduce_sum(fn)


