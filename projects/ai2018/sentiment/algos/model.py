#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ptr-net.py
#        \author   chenghuige  
#          \date   2018-01-15 11:50:08.306272
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf  
flags = tf.app.flags
FLAGS = flags.FLAGS

from tensorflow import keras

import wenzheng
from wenzheng.utils import vocabulary, embedding

from algos.config import NUM_CLASSES, NUM_ATTRIBUTES
from algos.weights import *

import melt
logging = melt.logging
import numpy as np

class Model(keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    ## adadelta adagrad will need cpu, so just use adam..
    #with tf.device('/cpu:0'):
    self.embedding = wenzheng.utils.Embedding(vocab_size, FLAGS.emb_dim, 
                                              FLAGS.word_embedding_file, 
                                              trainable=FLAGS.finetune_word_embedding)
    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.keep_prob = FLAGS.keep_prob

    self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)

    logging.info('encoder_output_method:', FLAGS.encoder_output_method, 'topk:', FLAGS.top_k)
    self.pooling = melt.layers.Pooling(FLAGS.encoder_output_method, top_k=FLAGS.top_k)
    #self.pooling = keras.layers.GlobalMaxPool1D()

    self.logits = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES, activation=None)

  def call(self, input, training=False):
    x = input['content']
    batch_size = melt.get_shape(x, 0)
    length = melt.length(x)
    #with tf.device('/cpu:0'):
    x = self.embedding(x)
    
    num_units = [melt.get_shape(x, -1) if layer == 0 else 2 * self.num_units for layer in range(self.num_layers)]
    #print('----------------length', tf.reduce_max(length), inputs.comment.shape)
    mask_fws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
    mask_bws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
    x = self.encode(x, length, mask_fws=mask_fws, mask_bws=mask_bws)
    #x = self.encode(x)
    x = self.pooling(x, length)
    #x = self.pooling(x)

    x = self.logits(x)

    if training and FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES * NUM_CLASSES:
      x = melt.adjust_lrs(x)

    x = tf.reshape(x, [batch_size, NUM_ATTRIBUTES, NUM_CLASSES])
    
    return x


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
      loss = melt.losses.focal_loss(y_, y)

  if FLAGS.na_ratio > 0.:
    y_ = tf.concat([y_[:,:,0:1], tf.reduce_sum(y_[:,:,1:], -1, keepdims=True)], -1)
    y = tf.one_hot(tf.to_int64(y > 0), 2)
    if no_weights():
      bloss = tf.losses.sigmoid_cross_entropy(y, y_)
    else:
      bloss = tf.losses.sigmoid_cross_entropy(y, y_, reduction=tf.losses.Reduction.NONE)
      bloss = tf.reduce_sum(bloss * tf.expand_dims(weights, -1), -1)
      bloss = tf.reduce_mean(bloss)
    if FLAGS.na_ratio_add:
      loss = loss + FLAGS.na_ratio * bloss
    else:
      loss = (1 - FLAGS.na_ratio) * loss + FLAGS.na_ratio * bloss

  return loss
