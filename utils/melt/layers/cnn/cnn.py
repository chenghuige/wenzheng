#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   cnn.py
#        \author   chenghuige  
#          \date   2018-02-18 12:26:23.284086
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sys, os

from official.transformer.model import model_utils

keras = tf.keras
layers = tf.keras.layers
Layer = layers.Layer

import melt

from melt.rnn import encode_outputs, OutputMethod


"""
Hierarchical ConvNet
"""
#https://arxiv.org/pdf/1705.02364.pdf
#https://github.com/facebookresearch/InferSent/blob/master/models.py

# This is simple can be cnn baseline, but easy to overfit


class ConvNet(melt.Model):
  def __init__(self, 
               num_layers, 
               num_filters, 
               keep_prob=1.0,
               kernel_sizes = [3] * 7,
               use_position_encoding=False,
               **kwargs):
      super(ConvNet, self).__init__(**kwargs)
      self.num_layers = num_layers
      self.keep_prob = keep_prob
      self.num_filters = num_filters
      self.kernel_sizes = kernel_sizes # might try [3, 5, 7, 9, 11, 13] ?
      self.use_position_encoding = use_position_encoding

      assert self.num_filters

      assert self.num_layers <= len(kernel_sizes)

      self.conv1ds = [layers.Conv1D(filters=num_filters, kernel_size=kernel_sizes[layer], padding='same', activation=tf.nn.relu) for layer in range(self.num_layers)]

  # seq_len for rnn compact 
  def call(self, seq, seq_len=None, masks=None, 
           output_method=OutputMethod.all, 
           training=False):
    if self.use_position_encoding:
      hidden_size = melt.get_shape(seq, -1)
      # Scale embedding by the sqrt of the hidden size
      seq *= hidden_size ** 0.5

      # Create binary array of size [batch_size, length]
      # where 1 = padding, 0 = not padding
      padding = tf.to_float(tf.sequence_mask(seq_len))

      # Set all padding embedding values to 0
      seq *= tf.expand_dims(padding, -1)

      pos_encoding = model_utils.get_position_encoding(
          tf.shape(seq)[1], tf.shape(seq)[-1])
      seq = seq + pos_encoding

    num_filters = self.num_filters
    seqs = [seq]
    #batch_size = melt.get_batch_size(seq)

    for layer in range(self.num_layers):
      if masks is None:
        seq_ = melt.dropout(seq, self.keep_prob, training)
      else:
        seq_ = seq * masks[layer]
      seq = self.conv1ds[layer](seq_)
      seqs.append(seq)
    
    outputs = tf.concat(seqs[1:], 2)
    # not do any dropout in convet just dropout outside 
    # if self.is_train and self.keep_prob < 1:
    #   outputs = tf.nn.dropout(outputs, self.keep_prob)

    # compact for rnn with sate return
    return melt.rnn.encode_outputs(outputs, seq_len, output_method)
  