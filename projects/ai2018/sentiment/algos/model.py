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

import melt
    
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

    self.pooling = melt.layers.MaxPooling()
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

    x = tf.reshape(x, [batch_size, NUM_ATTRIBUTES, NUM_CLASSES])
    
    return x

def criterion(model, x, y, training=False):
  y_ = model(x, training=training)
  y += 2 
  return tf.losses.sparse_softmax_cross_entropy(logits=y_, labels=y) 
