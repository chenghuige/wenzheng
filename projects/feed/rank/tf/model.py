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
    self.emb_dim = FLAGS.hidden_size
    if FLAGS.field_emb:
      self.field_emb = keras.layers.Embedding(FLAGS.field_dict_size + 1, FLAGS.hidden_size)
      self.emb_dim += FLAGS.hidden_size

    self.mult = keras.layers.Multiply()
    
    self.emb_activation = None
    if FLAGS.emb_activation:
      self.emb_activation = keras.layers.Activation(FLAGS.emb_activation)
      self.bias = K.variable(value=np.array([0.]))
    
    if not FLAGS.mlp_dims:
      self.mlp = None
    else:
      dims = [int(x) for x in FLAGS.mlp_dims.split(',')]
      self.mlp = melt.layers.Mlp(dims, activation=FLAGS.dense_activation, drop_rate=FLAGS.mlp_drop)

    act = FLAGS.dense_activation if FLAGS.deep_final_act else None
    self.dense = keras.layers.Dense(1, activation=act)

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

    if FLAGS.field_concat:
      num_fields = FLAGS.field_dict_size
      #x = tf.math.unsorted_segment_sum(x, fields, num_fields)      
      x = melt.unsorted_segment_sum_emb(x, fields, num_fields)
      # like [512, 100 * 50]
      x = K.reshape(x, [-1, num_fields * self.emb_dim])
    else:
      if FLAGS.pooling == 'allsum':
        x = K.sum(x, 1)
      else:
        #TODO add melt.SumPooling.. 
        assert FLAGS.index_addone, 'can not calc length for like 0,1,2,0,0,0'
        c_len = melt.length(ids)
        x = self.pooling(x, c_len)

    if self.emb_activation:
      x = self.emb_activation(x + self.bias)

    if self.mlp:
      x = self.mlp(x, training=training)
    
    x = self.dense(x)
    x = K.squeeze(x, -1)
    return x

class WideDeep(keras.Model):   
  def __init__(self):
    super(WideDeep, self).__init__()
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
