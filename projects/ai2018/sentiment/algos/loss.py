#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2018-09-17 20:34:23.281520
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os
from algos.weights import *

def criterion(model, x, y, training=False):
  y_ = model(x, training=training)
  y += 2
  weights = get_weights(FLAGS.aspect, FLAGS.attr_index)
  
  #print(y_, y, weights)
  if training and FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
    assert FLAGS.loss == 'cross'
    loss = tf.losses.sparse_softmax_cross_entropy(logits=y_, labels=y, weights=weights, reduction=tf.losses.Reduction.NONE)
    loss = melt.adjust_lrs(loss)
    loss = tf.reduce_mean(loss)
  else: 
    if FLAGS.loss == 'cross':
      loss = tf.losses.sparse_softmax_cross_entropy(logits=y_, labels=y, weights=weights) 
    elif FLAGS.loss == 'focal':
      loss = melt.losses.focal_loss(y, y_)

  if FLAGS.na_loss_ratio > 0.:
    y_ = tf.concat([y_[:,:,0:1], tf.reduce_sum(y_[:,:,1:], -1, keepdims=True)], -1)
    y_onehot = tf.one_hot(tf.to_int64(y > 0), 2)
    if no_weights():
      bloss = tf.losses.sigmoid_cross_entropy(y_onehot, y_)
    else:
      bloss = tf.losses.sigmoid_cross_entropy(y_onehot, y_, reduction=tf.losses.Reduction.NONE)
      bloss = tf.reduce_sum(bloss * tf.expand_dims(weights, -1), -1)
      bloss = tf.reduce_mean(bloss)
    if FLAGS.na_loss_ratio_add:
      loss = loss + FLAGS.na_loss_ratio * bloss
    else:
      loss = (1 - FLAGS.na_loss_ratio) * loss + FLAGS.na_loss_ratio * bloss

  if FLAGS.earth_mover_loss_ratio > 0.:
    y_onehot = tf.one_hot(y, 4)
    earth_mover_loss = melt.losses.earth_mover_loss(y_onehot[:,:,1:], y_[:,:,1:])
    loss = loss + FLAGS.earth_mover_loss_ratio * earth_mover_loss

  return loss

