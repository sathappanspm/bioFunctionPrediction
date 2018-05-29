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
import numpy as np


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)

        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def calc_performance_metrics(labels, predictions, threshold):
    labels = tf.cast(labels, tf.bool)
    predictions = predictions > threshold
    tp = tf.reduce_sum(tf.cast(tf.logical_and(labels, predictions), tf.float32), axis=1)
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(labels), predictions), tf.float32), axis=1)
    fn = tf.reduce_sum(tf.cast(tf.logical_and(labels, tf.logical_not(predictions)), tf.float32), axis=1)
    precision = tp / (tp + fp + tf.constant(1e-7))
    recall = tp / (tp + fn + tf.constant(1e-7))
    f1 = 2 * (precision * recall) / (precision + recall + tf.constant(1e-7))
    return (
        tf.reduce_mean(precision, name='precision'),
        tf.reduce_mean(recall, name='recall'),
        tf.reduce_mean(f1, name='f1')
        )


def numpy_calc_performance_metrics(labels, predictions, threshold):
    labels = labels.astype(np.bool)
    predictions = predictions > threshold
    tp = np.sum(labels & predictions, axis=1)
    fp = np.sum(~labels & predictions, axis=1)
    fn = np.sum(labels & ~(predictions), axis=1)
    precision = tp / (tp + fp + (1e-7))
    recall = tp / (tp + fn + (1e-7))
    f1 = 2 * (precision * recall) / (precision + recall +(1e-7))
    return (
        np.mean(precision),
        np.mean(recall),
        np.mean(f1)
        )


def full_eval_matrix(labels, predictions):
    precision = []
    recall = []
    f1 = []
    for i in range(0.1, 0.9):
        p, r, f = calc_performance_metrics(labels, predictions, threshold)
        precision.append(p)
        recall.append(r)
        f1.append(f1)

    precision = tf.concat(precision)
    recall = tf.concat(recall)
    f1 = tf.concat(recall)
    return precision, recall, f1


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)

        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


