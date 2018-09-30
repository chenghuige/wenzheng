#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   baseline.py
#        \author   chenghuige  
#          \date   2018-09-28 15:16:36.730041
#   \Description  
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import sys 
import os

from tensorflow import keras

import wenzheng
from wenzheng.utils import vocabulary

from algos.config import NUM_CLASSES

import melt
logging = melt.logging
    
class Model(melt.Model):
  def __init__(self):
    super(Model, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    ## adadelta adagrad will need cpu, so just use adam..
    #with tf.device('/cpu:0'):
    self.embedding = wenzheng.Embedding(vocab_size, 
                                        FLAGS.emb_dim, 
                                        FLAGS.word_embedding_file, 
                                        trainable=FLAGS.finetune_word_embedding,
                                        vocab2_size=FLAGS.unk_vocab_size)
    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.keep_prob = FLAGS.keep_prob

    self.encode = wenzheng.Encoder(FLAGS.encoder_type)

    self.pooling = melt.layers.Pooling(
                          FLAGS.encoder_output_method, 
                          top_k=FLAGS.top_k, 
                          att_activation=getattr(tf.nn, FLAGS.att_activation))

    self.logits = keras.layers.Dense(NUM_CLASSES)
    self.logits2 = keras.layers.Dense(NUM_CLASSES)

  def call(self, input, training=False):
    x = input['rcontent'] if FLAGS.rcontent else input['content']
    #print(x.shape)
    batch_size = melt.get_shape(x, 0)
    length = melt.length(x)
    #with tf.device('/cpu:0'):
    x = self.embedding(x)
    
    x = self.encode(x, length, training=training)
    
    # must mask pooling when eval ? but seems much worse result
    #if not FLAGS.mask_pooling and training:
    if not FLAGS.mask_pooling:
      length = None
    x = self.pooling(x, length)

    if FLAGS.use_type:
      x = tf.concat([x, tf.expand_dims(tf.to_float(input['type']), 1)], 1)

    if not FLAGS.split_type:
      x = self.logits(x)
    else:
      x1 = self.logits(x)
      x2 = self.logits2(x)
      x = tf.cond(tf.cast(input['type'] == 0, tf.bool), lambda: (x1 + x2) / 2., lambda: x2)
    
    return x

class Model2(melt.Model):
  """
  same as Model but with bi encode separately for passage and query
  """
  def __init__(self):
    super(Model2, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    ## adadelta adagrad will need cpu, so just use adam..
    #with tf.device('/cpu:0'):
    self.embedding = wenzheng.Embedding(vocab_size, 
                                              FLAGS.emb_dim, 
                                              FLAGS.word_embedding_file, 
                                              trainable=FLAGS.finetune_word_embedding,
                                              vocab2_size=FLAGS.unk_vocab_size)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.keep_prob = FLAGS.keep_prob

    self.encode = melt.layers.CudnnRnn2(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)

    self.pooling = melt.layers.MaxPooling2()
    #self.pooling = keras.layers.GlobalMaxPool1D()

    self.logits = keras.layers.Dense(NUM_CLASSES, activation=None)
    self.logits2 = keras.layers.Dense(NUM_CLASSES, activation=None)

  def call(self, input, training=False):
    x1 = input['query']
    x2 = input['passage']
    length1 = melt.length(x1)
    length2 = melt.length(x2)
    #with tf.device('/cpu:0'):
    x1 = self.embedding(x1)
    x2 = self.embedding(x2)
    
    x = x1
    batch_size = melt.get_shape(x1, 0)

    num_units = [melt.get_shape(x, -1) if layer == 0 else 2 * self.num_units for layer in range(self.num_layers)]
    #print('----------------length', tf.reduce_max(length), inputs.comment.shape)
    mask_fws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
    mask_bws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
    
    x = self.encode(x1, length1, x2, length2, mask_fws=mask_fws, mask_bws=mask_bws)
    x = self.pooling(x, length1, length2)
    #x = self.pooling(x)

    if FLAGS.use_type:
      x = tf.concat([x, tf.expand_dims(tf.to_float(input['type']), 1)], 1)

    if not FLAGS.split_type:
      x = self.logits(x)
    else:
      x1 = self.logits(x)
      x2 = self.logits2(x)
      x = tf.cond(tf.cast(input['type'] == 0, tf.bool), lambda: (x1 + x2) / 2., lambda: x2)
    
    return x

  
