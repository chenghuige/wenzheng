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
import sys

import sys, os
import melt

#1 -------------https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/text_classification_cnn.py
#may be change the interfance to encode(inputs) ?
#conv2d too much mem to consume!
# this is text_cnn! TODO

def encode(word_vectors, seq_len=None, output_method='all'):
  """2 layer ConvNet to predict from sequence of words to a class."""
  ## output last dim is only N_FILTERS as tf demo default 10 so small for encoding information of text ?
  ## also if set large will easy to OOM I have limited length to 500
  N_FILTERS = 10 #filters is as output
  WINDOW_SIZE = 3
  #FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
  FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
  POOLING_WINDOW = 4
  POOLING_STRIDE = 2

  #[batch_size, length, emb_dim]
  word_vectors = tf.expand_dims(word_vectors, -1)
  emb_dim = word_vectors.get_shape()[-1]
  FILTER_SHAPE1 = [WINDOW_SIZE, emb_dim]
  with tf.variable_scope('cnn_layer1'):
    # Apply Convolution filtering on input sequence.
    conv1 = tf.layers.conv2d(
        word_vectors,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        # Add a ReLU for non linearity.
        activation=tf.nn.relu)
    print('-----conv1', conv1)
    # Max pooling across output of Convolution+Relu.
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    print('-----pool1', pool1)
    # Transpose matrix so that n_filters from convolution becomes width.
    pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    print('-----pool1', pool1)
  with tf.variable_scope('cnn_layer2'):
    # Second level of convolution filtering.
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID')
    print('-----conv2', conv2)
    # Max across each filter to get useful features for classification.
    #pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
    #print('-----pool2', pool2)
    #return pool2
    output = tf.squeeze(conv2, 2)
    print('--------output', output)
    #return tf.layers.dense(pool2, emb_dim)
    #return melt.rnn.encode_outputs(conv2, sequence_length=seq_len, output_method=output_method)
    return output