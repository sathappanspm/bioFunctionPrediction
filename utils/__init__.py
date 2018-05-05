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
