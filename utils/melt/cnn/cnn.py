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
import melt

from official.transformer.model import model_utils


"""
Hierarchical ConvNet
"""
#https://arxiv.org/pdf/1705.02364.pdf
#https://github.com/facebookresearch/InferSent/blob/master/models.py

# This is simple can be cnn baseline, but easy to overfit

class ConvNet(object):
  def __init__(self, num_layers, num_filters, 
               use_position_encoding=False, keep_prob=1.0, 
               is_train=None, scope="conv_net"):
      self.num_layers = num_layers
      self.keep_prob = keep_prob
      self.num_filters = num_filters
      self.is_train = is_train
      self.use_position_encoding = use_position_encoding
      self.scope = scope

      assert self.num_filters

  # seq_len for rnn compact 
  def encode(self, seq, seq_len=None, output_method='all'):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
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
     
      #kernel_sizes = [3, 5, 7, 9, 11, 13]
      kernel_sizes = [3] * 7
      assert self.num_layers <= len(kernel_sizes)

      for layer in range(self.num_layers):
        #input_size_ = melt.get_shape(seq, -1) if layer == 0 else num_filters
        seq = melt.dropout(seq, self.keep_prob, self.is_train)
        seq = tf.layers.conv1d(seqs[-1], num_filters, kernel_size=kernel_sizes[layer], padding='same', activation=tf.nn.relu)
        # mask = melt.dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
        #                   keep_prob=self.keep_prob, is_train=self.is_train, mode=None)
        #seq = tf.layers.conv1d(seqs[-1] * mask, num_filters, kernel_size=3, padding='same', activation=tf.nn.relu)
        #seq = tf.layers.conv1d(seqs[-1] * mask, num_filters, kernel_size=kernel_sizes[layer], padding='same', activation=tf.nn.relu)
        
        # if self.is_train and self.keep_prob < 1:
        #   seq = tf.nn.dropout(seq, self.keep_prob)
        #seq = melt.layers.batch_norm(seq, self.is_train, name='layer_%d' % layer)
        seqs.append(seq)
      
      outputs = tf.concat(seqs[1:], 2)
      # not do any dropout in convet just dropout outside 
      # if self.is_train and self.keep_prob < 1:
      #   outputs = tf.nn.dropout(outputs, self.keep_prob)

      # compact for rnn with sate return
      return melt.rnn.encode_outputs(outputs, seq_len, output_method)

class ConvNet2(object):
  def __init__(self, num_layers, num_units, keep_prob=1.0, is_train=None, scope="conv_net"):
      self.num_layers = num_layers
      self.keep_prob = keep_prob
      self.num_units = num_units
      self.is_train = is_train
      self.scope = scope

  # seq_len for rnn compact 
  def encode(self, seq, seq_len=None, output_method='all'):
    with tf.variable_scope(self.scope):
      num_filters = self.num_units
      seqs = [seq]
      batch_size = melt.get_batch_size(seq)
     
      kernel_sizes = [3, 5, 7, 9, 11, 13]
      #kernel_sizes = [3] * 7
      assert self.num_layers <= len(kernel_sizes)

      for layer in range(self.num_layers):
        input_size_ = melt.get_shape(seq, -1) if layer == 0 else num_filters
        seq = melt.dropout(seq, self.keep_prob, self.is_train)
        seq = tf.layers.conv1d(seqs[-1], num_filters, kernel_size=kernel_sizes[layer], padding='same', activation=tf.nn.relu)
        # mask = melt.dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
        #                   keep_prob=self.keep_prob, is_train=self.is_train, mode=None)
        #seq = tf.layers.conv1d(seqs[-1] * mask, num_filters, kernel_size=3, padding='same', activation=tf.nn.relu)
        #seq = tf.layers.conv1d(seqs[-1] * mask, num_filters, kernel_size=kernel_sizes[layer], padding='same', activation=tf.nn.relu)
        
        # if self.is_train and self.keep_prob < 1:
        #   seq = tf.nn.dropout(seq, self.keep_prob)
        #seq = melt.layers.batch_norm(seq, self.is_train, name='layer_%d' % layer)
        seqs.append(seq)
      
      outputs = tf.concat(seqs[1:], 2)
      # not do any dropout in convet just dropout outside 
      # if self.is_train and self.keep_prob < 1:
      #   outputs = tf.nn.dropout(outputs, self.keep_prob)

      # compact for rnn with sate return
      return melt.rnn.encode_outputs(outputs, seq_len, output_method)

