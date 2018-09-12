#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2018-09-02 10:24:27.910985
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

from algos.config import NUM_CLASSES

import melt
    
class MyModel(keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    # TODO tf bug ? below not save CuDNNGRU but ok if self.encodes = [CuDNNGRU]
    self.encodes =  [(keras.layers.CuDNNGRU(units=100, 
                                        return_sequences=True, 
                                        return_state=False, 
                                        recurrent_initializer='glorot_uniform'),)]
    self.abc = self.add_variable("abc", [1, 100], initializer=tf.ones_initializer(dtype=tf.float32))


  def call(self, x):
    return self.encodes[0][0](x) + self.abc

class Model(keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    #self.embedding = keras.layers.Embedding(vocab_size, FLAGS.emb_dim)
    #with tf.device('/cpu:0'):
    self.embedding = wenzheng.utils.Embedding(vocab_size, FLAGS.emb_dim, 
                                              FLAGS.word_embedding_file, 
                                              trainable=FLAGS.finetune_word_embedding)
    
    #self.encode = MyLayer()
    self.num_layers = 1
    self.num_units = FLAGS.rnn_hidden_size
    self.keep_prob = 0.7
    self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)

    # self.encode = keras.layers.CuDNNGRU(units=FLAGS.rnn_hidden_size, 
    # #self.encode = keras.layers.CuDNNLSTM(units=FLAGS.rnn_hidden_size, 
    #                                     return_sequences=True, 
    #                                     return_state=False, 
    #                                     recurrent_initializer='glorot_uniform')

    #self.encode = keras.layers.GRU(units=FLAGS.rnn_hidden_size, 
    #                     return_sequences=True, 
    #                     return_state=False, 
    #                     recurrent_activation='sigmoid', 
    #                     recurrent_initializer='glorot_uniform')

    #self.pooling = keras.layers.GlobalMaxPool1D()
    self.pooling = melt.layers.MaxPooling()

    self.logits = keras.layers.Dense(NUM_CLASSES, activation=None)

    self.temp = MyModel()

  def call(self, x, training=False):
    x = x['comment']
    batch_size = melt.get_shape(x, 0)
    length = melt.length(x)
    #with tf.device('/cpu:0'):
    x = self.embedding(x)
    
    num_units = [melt.get_shape(x, -1) if layer == 0 else 2 * self.num_units for layer in range(self.num_layers)]
    #print('----------------length', tf.reduce_max(length), inputs.comment.shape)
    mask_fws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
    mask_bws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
    #x = self.encode(x, length, mask_fws=mask_fws, mask_bws=mask_bws)
    x = self.encode(x)

    x = self.pooling(x, length)
    #x = self.pooling(x)
    x = self.logits(x)
    return x

def criterion(model, x, y, training=False):
  y_ = model(x, training=training)
  return tf.losses.sigmoid_cross_entropy(y, y_)   
