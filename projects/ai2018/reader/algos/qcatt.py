#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   qcatt.py
#        \author   chenghuige  
#          \date   2018-09-28 15:19:25.674053
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
  
class QCAttention(melt.Model):
  def __init__(self):
    super(QCAttention, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    self.embedding = wenzheng.Embedding(vocab_size, 
                                              FLAGS.emb_dim, 
                                              FLAGS.word_embedding_file, 
                                              trainable=FLAGS.finetune_word_embedding,
                                              vocab2_size=FLAGS.unk_vocab_size)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.keep_prob = FLAGS.keep_prob

    self.encode = wenzheng.Encoder(FLAGS.encoder_type)

    self.att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)


    self.att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner)
    self.pooling = melt.layers.MaxPooling()

    self.logits = keras.layers.Dense(NUM_CLASSES, activation=None)
    self.logits2 = keras.layers.Dense(NUM_CLASSES, activation=None)

  def call(self, input, training=False):
    q = input['query']
    c = input['passage']
    q_len = melt.length(q)
    c_len = melt.length(c)
    q_mask = tf.cast(q, tf.bool)
    q_emb = self.embedding(q)
    c_emb = self.embedding(c)
    
    x = c_emb
    batch_size = melt.get_shape(x, 0)

    num_units = [melt.get_shape(x, -1) if layer == 0 else 2 * self.num_units for layer in range(self.num_layers)]
    mask_fws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
    mask_bws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
    
    c = self.encode(c_emb, c_len, mask_fws=mask_fws, mask_bws=mask_bws)
    q = self.encode(q_emb, q_len, mask_fws=mask_fws, mask_bws=mask_bws)

    qc_att = self.att_dot_attention(c, q, mask=q_mask, training=training)

    num_units = [melt.get_shape(qc_att, -1) if layer == 0 else 2 * self.num_units for layer in range(self.num_layers)]
    mask_fws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(1)]
    mask_bws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(1)]
    x = self.att_encode(qc_att, c_len, mask_fws=mask_fws, mask_bws=mask_bws)

    x = self.pooling(x, c_len)

    if FLAGS.use_type:
      x = tf.concat([x, tf.expand_dims(tf.to_float(input['type']), 1)], 1)

    if not FLAGS.split_type:
      x = self.logits(x)
    else:
      x1 = self.logits(x)
      x2 = self.logits2(x)
      x = tf.cond(tf.cast(input['type'] == 0, tf.bool), lambda: (x1 + x2) / 2., lambda: x2)
    
    return x