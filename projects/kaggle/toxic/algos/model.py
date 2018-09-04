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

from wenzheng.utils import vocabulary, embedding

from algos.config import NUM_CLASSES

from dataset import Input

import melt
    
class Model(keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    self.embedding = keras.layers.Embedding(vocab_size, FLAGS.emb_dim)
    self.pooling = keras.layers.GlobalMaxPool1D()
    self.logits = keras.layers.Dense(NUM_CLASSES, activation=None)
    self.probs = keras.layers.Dense(NUM_CLASSES, activation=tf.nn.sigmoid)

  def call(self, inputs, training=False):
    #inputs = Input(*inputs)
    x = inputs.comment
    x = self.embedding(x)
    #x = self.pooling(x)
    x = melt.max_pooling(x, melt.length(inputs.comment))
    x = self.logits(x)
    return x


def calc_loss(model, inputs, training=False):
  #print('----', x)
  y_ = model(inputs, training=training)
  y = inputs.classes
  #y_ = model(x.comment)
  #y_ = model(tf.constant([[1,2,3,0], [2,2,3,4]]))
  return tf.losses.sigmoid_cross_entropy(y, y_)   

