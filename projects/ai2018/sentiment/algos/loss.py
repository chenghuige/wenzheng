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
from algos.config import NUM_CLASSES

#from evaluate import load_class_weights

def calc_loss(y, y_, weights, training=False):
  #y += 2
  #print(y_, y, weights)
  #-----------deprciated seems per class learning rate decay do not improve
  if training and FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
    assert FLAGS.loss == 'cross'
    loss = tf.losses.sparse_softmax_cross_entropy(logits=y_, labels=y, weights=weights, reduction=tf.losses.Reduction.NONE)
    loss = melt.adjust_lrs(loss)
    loss = tf.reduce_mean(loss)
    #print('--------------weights', weights)

    # if weights == 1:
    #   weights = tf.ones([FLAGS.num_learning_rate_weights], dtype=tf.float32)
    #weights = tf.expand_dims(weights *  tf.get_collection('learning_rate_weights')[-1], 0)
    #  FIXME weights actually is per example adjust not for classes.. should be of shape [batch_size]
    #loss = tf.losses.sparse_softmax_cross_entropy(logits=y_, labels=y, weights=weights)
  else: 
    if FLAGS.loss == 'cross':
      if not FLAGS.label_smoothing:
        loss = tf.losses.sparse_softmax_cross_entropy(y, y_, weights=weights) 
      else:
        onehot_labels = tf.one_hot(y, NUM_CLASSES)
        print('--------------using label smoothing', FLAGS.label_smoothing)
        loss = tf.losses.softmax_cross_entropy(onehot_labels, y_, weights=weights, label_smoothing=FLAGS.label_smoothing)
    elif FLAGS.loss == 'focal':
      loss = melt.losses.focal_loss(y, y_)

  #----------depreciated below has bug..
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

# now consider simple rule level 0 loss + level 1 loss
# not imporve, worse...
def calc_hier_loss(y, y_, weights):
  binary_label = tf.to_int64(tf.equal(y, 0))
  # sigmoid reduction by default is not None will return scalar and if set None will return result shape as label
  level0_loss  = tf.losses.sigmoid_cross_entropy(binary_label, y_[:,:,0], weights=weights, reduction=tf.losses.Reduction.NONE)
  mask = tf.to_float(1 - binary_label)
  # softmax loss reduction by defualt is not None and will return scalar, set None will return shape as label
  level1_loss = tf.losses.sparse_softmax_cross_entropy(tf.maximum(y - 1, 0), y_[:,:,1:], weights=weights, reduction=tf.losses.Reduction.NONE)

  loss = level0_loss + level1_loss * mask
  loss = tf.reduce_mean(loss)

  return loss

def calc_hier_neu_loss(y, y_, weights):
  binary_label = tf.to_int64(tf.equal(y, 0))
  # sigmoid reduction by default is not None will return scalar and if set None will return result shape as label
  level0_loss  = tf.losses.sigmoid_cross_entropy(binary_label, y_[:,:,0], weights=weights, reduction=tf.losses.Reduction.NONE)
  mask = tf.to_float(1 - binary_label)
  # softmax loss reduction by defualt is not None and will return scalar, set None will return shape as label
  level1_loss = tf.losses.sparse_softmax_cross_entropy(tf.maximum(y - 1, 0), y_[:,:,1:], weights=weights, reduction=tf.losses.Reduction.NONE)

  loss = level0_loss + level1_loss * mask

  # add neu binary
  cid = 2
  binary_label = tf.to_int64(tf.equal(y, cid))
  binary_loss = tf.losses.sigmoid_cross_entropy(binary_label, y_[:,:,cid], weights=weights, reduction=tf.losses.Reduction.NONE)
  loss = loss + binary_loss

  loss = tf.reduce_mean(loss)

  return loss

#now try to add neu binary loss, if imrove can try add all binary loss for na, neg,neu,pos
def calc_add_binary_loss(y, y_, cids, weights):
  reduction = tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS if FLAGS.loss_combine_by_scalar else tf.losses.Reduction.NONE
  loss = tf.losses.sparse_softmax_cross_entropy(logits=y_, labels=y, weights=weights, reduction=reduction) 
  for cid in cids:
    binary_label = tf.to_int64(tf.equal(y, cid))
    binary_loss = tf.losses.sigmoid_cross_entropy(binary_label, y_[:,:,cid], weights=weights, reduction=reduction)
    loss = loss + binary_loss * FLAGS.other_loss_factor
  loss = tf.reduce_mean(loss)
  return loss

def calc_binary_loss(y, y_, cid, weights):
  binary_label = tf.to_int64(tf.equal(y, cid))
  binary_loss = tf.losses.sigmoid_cross_entropy(binary_label, y_[:,:,cid], weights=weights)
  return binary_loss

def calc_regression_loss(y, y_, weights):
  y = y * 2 + 2
  return tf.losses.mean_squared_error(y, y_, weights=weights)

def calc_add_binaries_loss(y, y_, cid, weights):  
  loss = tf.losses.sparse_softmax_cross_entropy(logits=y_, labels=y, weights=weights, reduction=tf.losses.Reduction.NONE) 
  for cid in range(NUM_CLASSES):
    binary_label = tf.to_int64(tf.equal(y, cid))
    binary_loss = tf.losses.sigmoid_cross_entropy(binary_label, y_[:,:,cid], weights=weights, reduction=tf.losses.Reduction.NONE)
    loss = loss + binary_loss * (1 / NUM_CLASSES)
  loss = tf.reduce_mean(loss)
  return loss

def calc_binaries_only_loss(y, y_, cid, weights):  
  loss = None
  for cid in range(NUM_CLASSES):
    binary_label = tf.to_int64(tf.equal(y, cid))
    binary_loss = tf.losses.sigmoid_cross_entropy(binary_label, y_[:,:,cid], weights=weights, reduction=tf.losses.Reduction.NONE)
    if loss is None:
      loss = binary_loss
    else:
      loss = loss + binary_loss
    
  loss = tf.reduce_mean(loss)
  return loss
  
def criterion(model, x, y, training=False):
  y_ = model(x, training=training)
  weights = get_weights(FLAGS.aspect, FLAGS.attr_index)

  # only need this if we have label -1 means to mask 
  mask = y >= 0
  weights = weights * tf.to_float(mask)
  y = y * tf.to_int64(mask)

  # if FLAGS.use_class_weights:
  #   weights = load_class_weights()
  #   print('--------------', weights)
    
  if FLAGS.loss_type == 'normal':
    return calc_loss(y, y_, weights, training)
  elif FLAGS.loss_type == 'binary':
    return tf.losses.sigmoid_cross_entropy(tf.to_int64(y[:,FLAGS.binary_class_index:FLAGS.binary_class_index + 1] > 0), y_, 
                                           reduction=tf.losses.Reduction.NONE, weights=weights)
  elif FLAGS.loss_type == 'hier':
    # not improve deprecated
    return calc_hier_loss(y, y_, weights)
  elif FLAGS.loss_type == 'add_neu_binary':
    # neu cid is 2
    return calc_add_binary_loss(y, y_, [2], weights)
  elif FLAGS.loss_type.startswith('add_binary_'):
    cids = [int(x) for x in FLAGS.loss_type.split('_')[-1].split(',')]
    return calc_add_binary_loss(y, y_, cids, weights)
  elif FLAGS.loss_type.startswith('binary_'):
    cid = int(FLAGS.loss_type.split('_')[-1])
    return calc_binary_loss(y, y_, cid, weights)
  elif FLAGS.loss_type == 'regression':
    return calc_regression_loss(y, y_, weights)
  elif FLAGS.loss_type == 'add_binaries':
    return calc_add_binaries_loss(y, y_, weights)
  elif FLAGS.loss_type == 'binaries_only':
    return calc_binaries_only_loss(y, y_, weights)
  elif FLAGS.loss_type == 'hier_neu':
    return calc_hier_neu_loss(y, y_, weights)
  else:
    raise ValueError(f'Unsupported loss type{FLAGS.loss_type}')
