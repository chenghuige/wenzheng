#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2018-10-11 16:56:08.992105
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

from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
from tensorflow.contrib.eager.python import tfe

layers = tf.keras.layers

import melt
import wenzheng


class PTBModel(tf.keras.Model):
  """LSTM for word language modeling.
  Model described in:
  (Zaremba, et. al.) Recurrent Neural Network Regularization
  http://arxiv.org/abs/1409.2329
  See also:
  https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb
  """
  def __init__(self,
               vocab_size,
               embedding_dim,
               hidden_dim,
               num_layers,
               dropout_ratio,
               concat_layers=False,
               use_cudnn_rnn=True):
    super(PTBModel, self).__init__()

    self.keep_ratio = 1 - dropout_ratio
    self.use_cudnn_rnn = use_cudnn_rnn
    self.embedding = wenzheng.Embedding(vocab_size, embedding_dim, FLAGS.word_embedding_file)

    self.rnn = melt.layers.CudnnRnn(num_layers, hidden_dim, keep_prob=1 - dropout_ratio, 
                                    concat_layers=concat_layers, return_state=True, cell='gru')

    self.linear = layers.Dense(
        vocab_size, kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1))

    if not concat_layers:
      self._output_shape = [-1, 2 * hidden_dim]
    else:
      self._output_shape = [-1, 2 * hidden_dim * num_layers]

  def call(self, input_seq, training=False):
    """Run the forward pass of PTBModel.
    Args:
      input_seq: [length, batch] shape int64 tensor.
      training: Is this a training call.
    Returns:
      outputs tensors of inference.
    """
    #input_seq = tf.transpose(input_seq, [1,0])
    y = self.embedding(input_seq)
    if training:
      y = tf.nn.dropout(y, self.keep_ratio)
    y = self.rnn(y, training=training)[0]
    return self.linear(tf.reshape(y, self._output_shape))