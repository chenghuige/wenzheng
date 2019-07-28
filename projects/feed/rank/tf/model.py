#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2019-07-26 20:15:30.419843
#   \Description  TODO maybe input should be more flexible, signle feature, cross, cat, lianxu choumi
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import gezi
import melt

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

keras = tf.keras
from keras import backend as K

import numpy as np

# output logits!
class Wide(keras.Model):
  def __init__(self):
    super(Wide, self).__init__()
    self.emb = keras.layers.Embedding(FLAGS.feature_dict_size + 1, 1)
    if FLAGS.wide_addval:
      self.mult = keras.layers.Multiply()
    self.bias = K.variable(value=np.array([0.]))

  def call(self, input):
    ids = input['index']
    values = input['value']

    x = self.emb(ids)
    x = K.squeeze(x, -1)
    # strage, eval for wide only addval will got worse result
    if FLAGS.wide_addval:
      x = self.mult([x, values])
    x = K.sum(x,1)
    x = x + self.bias
    return x  

class Deep(keras.Model):
  def __init__(self):
    super(Deep, self).__init__()
    # # do not need two many deep embdding, only need some or cat not cross TODO
    # # STILL OOM FIXME...
    # if FLAGS.hidden_size > 50:
    #   print('---------------put emb on cpu')
    #   with tf.device('/cpu:0'):
    #     self.emb = keras.layers.Embedding(FLAGS.feature_dict_size + 1, FLAGS.hidden_size)
    # else:
    self.emb = keras.layers.Embedding(FLAGS.feature_dict_size + 1, FLAGS.hidden_size)
    if FLAGS.field_emb:
      self.field_emb = keras.layers.Embedding(FLAGS.field_dict_size + 1, FLAGS.hidden_size)
    self.mult = keras.layers.Multiply()
    
    self.emb_activation = None
    if FLAGS.emb_activation:
      self.emb_activation = keras.layers.Activation(FLAGS.emb_activation)
      self.bias = K.variable(value=np.array([0.]))
    
    self.dense = keras.layers.Dense(1, activation=FLAGS.dense_activation)

    if FLAGS.pooling != 'allsum':
      self.pooling = melt.layers.Pooling(FLAGS.pooling)
   
  def call(self, input, training=False):
    ids = input['index']
    values = input['value']
    fields = input['field']
    
    # if FLAGS.hidden_size > 50:
    #   with tf.device('/cpu:0'):
    #     x = self.emb(ids)
    # else:
    x = self.emb(ids)
    if FLAGS.field_emb:
      x = K.concatenate([x, self.field_emb(fields)], axis=-1)

    if FLAGS.deep_addval:
      values = K.expand_dims(values, -1)
      x = self.mult([x, values])

    if FLAGS.pooling == 'allsum':
     x = K.sum(x, 1)
    else:
      #TODO add melt.SumPooling.. 
      assert FLAGS.index_addone, 'can not calc length for like 0,1,2,0,0,0'
      c_len = melt.length(ids)
      x = self.pooling(x, c_len)

    if self.emb_activation:
      x = self.emb_activation(x + self.bias)
    x = self.dense(x)
    x = K.squeeze(x, -1)
    return x

# # Supporting filed embedding combine  TODO not correct
# class Deep2(keras.Model):
#   def __init__(self):
#     super(Deep, self).__init__()
#     self.emb = keras.layers.Embedding(FLAGS.feature_dict_size, FLAGS.hidden_size)
#     self.mult = keras.layers.Multiply()
    
#     self.emb_activation = None
#     if FLAGS.emb_activation:
#       self.emb_activation = keras.layers.Activation(FLAGS.emb_activation)
#       self.bias = K.variable(value=np.array([0.]))
    
#     self.dense = keras.layers.Dense(1, activation=FLAGS.dense_activation)
   
#   def call(self, input):
#     ids = input['index']
#     values = input['value']
#     fields = input['field']
    
#     batch_size = ids.shape[0]
#     length = ids.shape[1]

#     m = K.reshape(tf.range(batch_size), [batch_size, 1])
#     m = K.concat([m] * length, axis=1) * batch_size

#     x = self.emb(ids)
#     emb_dim = self.emb.shape[-1]
#     x = K.reshape(x, [-1, emb_dim])
#     fields = K.reshape(fields, [-1, emb_dim])
#     fields = fields + m

#     x = K.segment_sum(x, fields)
#     x = K.reshape(x, [batch_size, -1, emb_dim])

#     if FLAGS.deep_addval:
#       values = K.expand_dims(values, -1)
#       x = self.mult([x, values])

#     x = K.sum(x, 1)
#     if self.emb_activation:
#       x = self.emb_activation(x + self.bias)
#     x = self.dense(x)
#     x = K.squeeze(x, -1)
#     return x


# class WideDeep(keras.Model):   
#   def __init__(self):
#     super(WideDeep, self).__init__()
#     self.wide = Wide()
#     self.deep = Deep()

#   def call(self, input):
#     w = self.wide(input)
#     d = self.deep(input)
#     #------seems works bad then below WideDeep2
#     x = w + d
#     return x

class WideDeep(keras.Model):   
  def __init__(self):
    super(WideDeep2, self).__init__()
    self.wide = Wide()
    self.deep = Deep() 
    self.dense = keras.layers.Dense(1)

  def call(self, input, training=False):
    w = self.wide(input)
    d = self.deep(input, training=training)
    if FLAGS.deep_wide_combine == 'concat':
      x = K.stack([w, d], 1)
      x = self.dense(x)
      x = K.squeeze(x, -1)
    else:
      x = w + d
    return x

WideDeep2 = WideDeep