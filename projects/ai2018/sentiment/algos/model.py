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
from wenzheng.utils import vocabulary

from algos.config import NUM_CLASSES, NUM_ATTRIBUTES
from algos.weights import *
import prepare.config

import melt
logging = melt.logging
import gezi 

import numpy as np

#same as ModelV2 but this is pytorch Mreader like attention(light attention)
class ModelV3(melt.Model):
  def __init__(self):
    super(ModelV3, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    logging.info('vocab_size:', vocab_size)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size if FLAGS.encoder_type != 'convnet' else FLAGS.num_filters
    self.keep_prob = FLAGS.keep_prob

    if FLAGS.hop > 1:
      assert self.num_layers == 1 and FLAGS.att_combiner == 'sfu', 'mreader must set num layers to 1 so can iterative align if you set hop > 1, and use sfu as combiner'

    self.embedding = wenzheng.Embedding(vocab_size, 
                                        FLAGS.emb_dim, 
                                        FLAGS.word_embedding_file, 
                                        trainable=FLAGS.finetune_word_embedding)

    if FLAGS.use_label_emb or FLAGS.use_label_att:
      #assert not (FLAGS.use_label_emb and FLAGS.use_label_att)
      self.label_emb_height = NUM_CLASSES * NUM_ATTRIBUTES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      if FLAGS.use_label_emb and FLAGS.use_label_att:
        assert self.label_emb_height == NUM_CLASSES * NUM_ATTRIBUTES
      self.label_embedding = melt.layers.Embedding(self.label_emb_height, FLAGS.emb_dim)
      if FLAGS.use_label_emb:
        #self.label_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
        self.label_dense = keras.layers.Dense(FLAGS.emb_dim, use_bias=False)
      if FLAGS.use_label_att:
        self.att_dot_attentions = []
        self.att_encodes = []
        for _ in range(FLAGS.hop):
          self.att_dot_attentions.append(melt.layers.melt.layers.SeqAttnMatch(combiner=FLAGS.att_combiner, identity=True))
          #self.att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=0.5, combiner=FLAGS.att_combiner)
          self.att_encodes.append(melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob))

    #self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)
    self.encode = wenzheng.Encoder(FLAGS.encoder_type)
    
    #self.multiplier = 2 if self.encode.bidirectional else 1

    # hier a bit worse
    self.hier_encode = melt.layers.HierEncode() if FLAGS.use_hier_encode else None

    if FLAGS.use_self_match:
      self.match_dot_attentions = []
      self.match_encodes = []
      for _ in range(FLAGS.hop):
        self.match_dot_attentions.append(melt.layers.SelfAttnMatch(combiner=FLAGS.att_combiner, identity=True, diag=False))
        self.match_encodes.append(melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob))
      #self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)

    # top-k best, max,att can benfit ensemble(better then max, worse then topk-3), topk,att now best with 2layers
    logging.info('encoder_output_method:', FLAGS.encoder_output_method)
    logging.info('topk:', FLAGS.top_k)
    self.pooling = melt.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        top_k=FLAGS.top_k,
                        att_activation=getattr(tf.nn, FLAGS.att_activation))
    #self.pooling = keras.layers.GlobalMaxPool1D()

    # mlp not help much!
    if FLAGS.mlp_ratio != 0:
      self.dropout = keras.layers.Dropout(0.3)
      if FLAGS.mlp_ratio < 0:
        # here activation hurt perf!
        #self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2, activation=tf.nn.relu)
        self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2)
      elif FLAGS.mlp_ratio <= 1:
        self.dense = melt.layers.DynamicDense(FLAGS.mlp_ratio)
      else:
        self.dense = kears.layers.Dense(int(FLAGS.mlp_ratio))
    else:
      self.dense = None

    self.num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
    self.logits = keras.layers.Dense(NUM_ATTRIBUTES * self.num_classes, activation=None)
    
  def call(self, input, training=False):
    x = input['content'] 

    c_mask = tf.cast(x, tf.bool)
    batch_size = melt.get_shape(x, 0)
    c_len = melt.length(x)

    if FLAGS.rnn_no_padding:
      logging.info('------------------no padding! train or eval')
      c_len = tf.ones([batch_size], dtype=x.dtype) * tf.cast(melt.get_shape(x, -1), x.dtype)
      
    #with tf.device('/cpu:0'):
    x = self.embedding(x)

    x = self.encode(x, c_len, training=training)
    #x = self.encode(x)

    # not help
    if self.hier_encode is not None:
      x = self.hier_encode(x, c_len)

    # yes just using label emb..
    label_emb = self.label_embedding(None)
    label_seq = tf.tile(tf.expand_dims(label_emb, 0), [batch_size, 1, 1])
    label_seq = self.encode(label_seq, tf.ones([batch_size], dtype=tf.int64) * self.label_emb_height, training=training)

    for i in range(FLAGS.hop):
      x = self.att_dot_attentions[i](x, label_seq, mask=tf.ones([batch_size, self.label_emb_height], tf.bool), training=training)
      x = self.att_encodes[i](x, c_len, training=training)
      x = self.match_dot_attentions[i](x, mask=c_mask, training=training) 
      x = self.match_encodes[i](x, c_len, training=training) 

    x = self.pooling(x, c_len, calc_word_scores=self.debug)
    #x = self.pooling(x)

    # not help much
    if self.dense is not None:
      x = self.dense(x)
      x = self.dropout(x, training=training)

    if not FLAGS.use_label_emb:
      x = self.logits(x)
    else:
      x = self.label_dense(x)
      # TODO..
      x = melt.dot(x, self.label_embedding(None))

    # # No help match
    # if training and FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES * NUM_CLASSES:
    #   x = melt.adjust_lrs(x)

    x = tf.reshape(x, [batch_size, NUM_ATTRIBUTES, self.num_classes])
    
    return x

# same as Model but attention no dropout(RNet without attention dropout v3/gru.5k(ModelV2) similary result as gru.5k(Model) but trian loss lower)
class ModelV2(melt.Model):
  def __init__(self):
    super(ModelV2, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    logging.info('vocab_size:', vocab_size)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size if FLAGS.encoder_type != 'convnet' else FLAGS.num_filters
    self.keep_prob = FLAGS.keep_prob

    self.embedding = wenzheng.Embedding(vocab_size, 
                                        FLAGS.emb_dim, 
                                        FLAGS.word_embedding_file, 
                                        trainable=FLAGS.finetune_word_embedding)

    if FLAGS.use_label_emb or FLAGS.use_label_att:
      #assert not (FLAGS.use_label_emb and FLAGS.use_label_att)
      self.label_emb_height = NUM_CLASSES * NUM_ATTRIBUTES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      if FLAGS.use_label_emb and FLAGS.use_label_att:
        assert self.label_emb_height == NUM_CLASSES * NUM_ATTRIBUTES
      self.label_embedding = melt.layers.Embedding(self.label_emb_height, FLAGS.emb_dim)
      if FLAGS.use_label_emb:
        #self.label_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
        self.label_dense = keras.layers.Dense(FLAGS.emb_dim, use_bias=False)
      if FLAGS.use_label_att:
        self.att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, combiner=FLAGS.att_combiner)
        #self.att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=0.5, combiner=FLAGS.att_combiner)
        self.att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)

    #self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)
    self.encode = wenzheng.Encoder(FLAGS.encoder_type)
    
    #self.multiplier = 2 if self.encode.bidirectional else 1

    # hier a bit worse
    self.hier_encode = melt.layers.HierEncode() if FLAGS.use_hier_encode else None

    if FLAGS.use_self_match:
      self.match_dot_attention = melt.layers.DotAttention(hidden=self.num_units, combiner=FLAGS.att_combiner)
      self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)
      #self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)

    # top-k best, max,att can benfit ensemble(better then max, worse then topk-3), topk,att now best with 2layers
    logging.info('encoder_output_method:', FLAGS.encoder_output_method)
    logging.info('topk:', FLAGS.top_k)
    self.pooling = melt.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        top_k=FLAGS.top_k,
                        att_activation=getattr(tf.nn, FLAGS.att_activation))
    #self.pooling = keras.layers.GlobalMaxPool1D()

    # mlp not help much!
    if FLAGS.mlp_ratio != 0:
      self.dropout = keras.layers.Dropout(0.3)
      if FLAGS.mlp_ratio < 0:
        # here activation hurt perf!
        #self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2, activation=tf.nn.relu)
        self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2)
      elif FLAGS.mlp_ratio <= 1:
        self.dense = melt.layers.DynamicDense(FLAGS.mlp_ratio)
      else:
        self.dense = kears.layers.Dense(int(FLAGS.mlp_ratio))
    else:
      self.dense = None

    self.num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
    self.logits = keras.layers.Dense(NUM_ATTRIBUTES * self.num_classes, activation=None)
    
  def call(self, input, training=False):
    x = input['content'] 

    c_mask = tf.cast(x, tf.bool)
    batch_size = melt.get_shape(x, 0)
    c_len = melt.length(x)

    if FLAGS.rnn_no_padding:
      logging.info('------------------no padding! train or eval')
      c_len = tf.ones([batch_size], dtype=x.dtype) * tf.cast(melt.get_shape(x, -1), x.dtype)

    #with tf.device('/cpu:0'):
    x = self.embedding(x)

    x = self.encode(x, c_len, training=training)
    #x = self.encode(x)

    # not help
    if self.hier_encode is not None:
      x = self.hier_encode(x, c_len)

    if FLAGS.use_label_att:
      label_emb = self.label_embedding(None)
      label_seq = tf.tile(tf.expand_dims(label_emb, 0), [batch_size, 1, 1])
      x = self.att_dot_attention(x, label_seq, mask=tf.ones([batch_size, self.label_emb_height], tf.bool), training=training)

      if not FLAGS.simple_label_att:
        x = self.att_encode(x, c_len, training=training)

    # pust self match at last
    if FLAGS.use_self_match:
       x = self.match_dot_attention(x, x, mask=c_mask, training=training) 
       x = self.match_encode(x, c_len, training=training) 

    x = self.pooling(x, c_len, calc_word_scores=self.debug)
    #x = self.pooling(x)

    # not help much
    if self.dense is not None:
      x = self.dense(x)
      x = self.dropout(x, training=training)

    if not FLAGS.use_label_emb:
      x = self.logits(x)
    else:
      x = self.label_dense(x)
      # TODO..
      x = melt.dot(x, self.label_embedding(None))

    # # No help match
    # if training and FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES * NUM_CLASSES:
    #   x = melt.adjust_lrs(x)

    x = tf.reshape(x, [batch_size, NUM_ATTRIBUTES, self.num_classes])
    
    return x


#same as RNetV3 but all use sfu and since dot attention with sfu not good, change 
# V4 is bad , seems for label attention can only use gate
class RNetV4(melt.Model):
  def __init__(self):
    super(RNetV4, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    logging.info('vocab_size:', vocab_size)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size if FLAGS.encoder_type != 'convnet' else FLAGS.num_filters
    self.keep_prob = FLAGS.keep_prob

    self.embedding = wenzheng.Embedding(vocab_size, 
                                        FLAGS.emb_dim, 
                                        FLAGS.word_embedding_file, 
                                        trainable=FLAGS.finetune_word_embedding)

    if FLAGS.use_label_emb or FLAGS.use_label_att:
      #assert not (FLAGS.use_label_emb and FLAGS.use_label_att)
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      if FLAGS.use_label_emb and FLAGS.use_label_att:
        assert self.label_emb_height == NUM_CLASSES * NUM_ATTRIBUTES
      self.label_embedding = melt.layers.Embedding(self.label_emb_height, FLAGS.emb_dim)
      if FLAGS.use_label_emb:
        #self.label_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
        self.label_dense = keras.layers.Dense(FLAGS.emb_dim)
      if FLAGS.use_label_att:
        self.att_dot_attention = melt.layers.melt.layers.SeqAttnMatch(combiner='sfu', identity=True, keep_prob=self.keep_prob)
        #self.att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=0.5, combiner=FLAGS.att_combiner)
        self.att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)
      self.label_encode = keras.layers.Dense(2 * self.num_units * self.num_layers)

    #self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)
    self.encode = wenzheng.Encoder(FLAGS.encoder_type)
    
    #self.multiplier = 2 if self.encode.bidirectional else 1

    # hier a bit worse
    self.hier_encode = melt.layers.HierEncode() if FLAGS.use_hier_encode else None

    if FLAGS.use_self_match:
      self.match_dot_attention = melt.layers.SelfAttnMatch(combiner='sfu', identity=True, diag=False, keep_prob=self.keep_prob)
      self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)
      #self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)

    # top-k best, max,att can benfit ensemble(better then max, worse then topk-3), topk,att now best with 2layers
    logging.info('encoder_output_method:', FLAGS.encoder_output_method)
    logging.info('topk:', FLAGS.top_k)
    self.pooling = melt.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        top_k=FLAGS.top_k,
                        att_activation=getattr(tf.nn, FLAGS.att_activation))
    #self.pooling = keras.layers.GlobalMaxPool1D()

    # mlp not help much!
    if FLAGS.mlp_ratio != 0:
      self.dropout = keras.layers.Dropout(0.3)
      if FLAGS.mlp_ratio < 0:
        # here activation hurt perf!
        #self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2, activation=tf.nn.relu)
        self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2)
      elif FLAGS.mlp_ratio <= 1:
        self.dense = melt.layers.DynamicDense(FLAGS.mlp_ratio)
      else:
        self.dense = kears.layers.Dense(int(FLAGS.mlp_ratio))
    else:
      self.dense = None

    self.num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
    self.logits = keras.layers.Dense(NUM_ATTRIBUTES * self.num_classes, activation=None)
    
  def call(self, input, training=False):
    x = input['content'] 

    c_mask = tf.cast(x, tf.bool)
    batch_size = melt.get_shape(x, 0)
    c_len = melt.length(x)

    if FLAGS.rnn_no_padding:
      logging.info('------------------no padding! train or eval')
      c_len = tf.ones([batch_size], dtype=x.dtype) * tf.cast(melt.get_shape(x, -1), x.dtype)

    #with tf.device('/cpu:0'):
    x = self.embedding(x)

    x = self.encode(x, c_len, training=training)
    #x = self.encode(x)

    # not help
    if self.hier_encode is not None:
      x = self.hier_encode(x, c_len)

    if FLAGS.use_label_att:
      label_emb = self.label_embedding(None)
      label_seq = tf.tile(tf.expand_dims(label_emb, 0), [batch_size, 1, 1])
      label_seq = self.label_encode(label_seq)
      x = self.att_dot_attention(x, label_seq, mask=tf.ones([batch_size, self.label_emb_height], tf.bool), training=training)

      if not FLAGS.simple_label_att:
        x = self.att_encode(x, c_len, training=training)

    # pust self match at last
    if FLAGS.use_self_match:
        x = self.match_dot_attention(x, mask=c_mask, training=training) 
        x = self.match_encode(x, c_len, training=training) 

    x = self.pooling(x, c_len, calc_word_scores=self.debug)
    #x = self.pooling(x)

    # not help much
    if self.dense is not None:
      x = self.dense(x)
      x = self.dropout(x, training=training)

    if not FLAGS.use_label_emb:
      x = self.logits(x)
    else:
      x = self.label_dense(x)
      # TODO..
      x = melt.dot(x, self.label_embedding(None))

    # # No help match
    # if training and FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES * NUM_CLASSES:
    #   x = melt.adjust_lrs(x)

    x = tf.reshape(x, [batch_size, NUM_ATTRIBUTES, self.num_classes])
    
    return x

# same as RNetV2 but is gate + sfu
class RNetV3(melt.Model):
  def __init__(self):
    super(RNetV3, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    logging.info('vocab_size:', vocab_size)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size if FLAGS.encoder_type != 'convnet' else FLAGS.num_filters
    self.keep_prob = FLAGS.keep_prob

    self.embedding = wenzheng.Embedding(vocab_size, 
                                        FLAGS.emb_dim, 
                                        FLAGS.word_embedding_file, 
                                        trainable=FLAGS.finetune_word_embedding)

    if FLAGS.use_label_emb or FLAGS.use_label_att:
      #assert not (FLAGS.use_label_emb and FLAGS.use_label_att)
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      if FLAGS.use_label_emb and FLAGS.use_label_att:
        assert self.label_emb_height == NUM_CLASSES * NUM_ATTRIBUTES
      self.label_embedding = melt.layers.Embedding(self.label_emb_height, FLAGS.emb_dim)
      if FLAGS.use_label_emb:
        #self.label_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
        self.label_dense = keras.layers.Dense(FLAGS.emb_dim, use_bias=False)
      if FLAGS.use_label_att:
        self.att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner='gate')
        #self.att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=0.5, combiner=FLAGS.att_combiner)
        self.att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)
      if FLAGS.use_label_rnn:
        self.label_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)

    #self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)
    self.encode = wenzheng.Encoder(FLAGS.encoder_type)
    
    #self.multiplier = 2 if self.encode.bidirectional else 1

    # hier a bit worse
    self.hier_encode = melt.layers.HierEncode() if FLAGS.use_hier_encode else None

    if FLAGS.use_self_match:
      self.match_dot_attention = melt.layers.SelfAttnMatch(combiner='sfu', identity=True, diag=False, keep_prob=self.keep_prob)
      self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)
      #self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)
 
    # top-k best, max,att can benfit ensemble(better then max, worse then topk-3), topk,att now best with 2layers
    logging.info('encoder_output_method:', FLAGS.encoder_output_method)
    logging.info('topk:', FLAGS.top_k)
    self.pooling = melt.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        top_k=FLAGS.top_k,
                        att_activation=getattr(tf.nn, FLAGS.att_activation))
    #self.pooling = keras.layers.GlobalMaxPool1D()

    # mlp not help much!
    if FLAGS.mlp_ratio != 0:
      self.dropout = keras.layers.Dropout(0.3)
      if FLAGS.mlp_ratio < 0:
        # here activation hurt perf!
        #self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2, activation=tf.nn.relu)
        self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2)
      elif FLAGS.mlp_ratio <= 1:
        self.dense = melt.layers.DynamicDense(FLAGS.mlp_ratio)
      else:
        self.dense = kears.layers.Dense(int(FLAGS.mlp_ratio))
    else:
      self.dense = None

    self.num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
    self.logits = keras.layers.Dense(NUM_ATTRIBUTES * self.num_classes, activation=None)
    
  def call(self, input, training=False):
    x = input['content'] 

    c_mask = tf.cast(x, tf.bool)
    batch_size = melt.get_shape(x, 0)
    c_len = melt.length(x)

    if FLAGS.rnn_no_padding:
      logging.info('------------------no padding! train or eval')
      c_len = tf.ones([batch_size], dtype=x.dtype) * tf.cast(melt.get_shape(x, -1), x.dtype)

    #with tf.device('/cpu:0'):
    x = self.embedding(x)

    x = self.encode(x, c_len, training=training)
    #x = self.encode(x)

    # not help
    if self.hier_encode is not None:
      x = self.hier_encode(x, c_len)

    if FLAGS.use_label_att:
      label_emb = self.label_embedding(None)
      label_seq = tf.tile(tf.expand_dims(label_emb, 0), [batch_size, 1, 1])
      if FLAGS.use_label_rnn:
        label_seq = self.label_encode(label_seq, tf.ones([batch_size], tf.int32) * tf.cast(melt.get_shape(label_emb, 1), tf.int32))
      x = self.att_dot_attention(x, label_seq, mask=tf.ones([batch_size, self.label_emb_height], tf.bool), training=training)

      if not FLAGS.simple_label_att:
        x = self.att_encode(x, c_len, training=training)

    # pust self match at last
    if FLAGS.use_self_match:
       x = self.match_dot_attention(x, mask=c_mask, training=training) 
       x = self.match_encode(x, c_len, training=training) 

    x = self.pooling(x, c_len, calc_word_scores=self.debug)
    #x = self.pooling(x)

    # not help much
    if self.dense is not None:
      x = self.dense(x)
      x = self.dropout(x, training=training)

    if not FLAGS.use_label_emb:
      x = self.logits(x)
    else:
      x = self.label_dense(x)
      # TODO..
      x = melt.dot(x, self.label_embedding(None))

    # # No help match
    # if training and FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES * NUM_CLASSES:
    #   x = melt.adjust_lrs(x)

    x = tf.reshape(x, [batch_size, NUM_ATTRIBUTES, self.num_classes])
    
    return x

# same as Model but for math attention using SeqAttn
class RNetV2(melt.Model):
  def __init__(self):
    super(RNetV2, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    logging.info('vocab_size:', vocab_size)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size if FLAGS.encoder_type != 'convnet' else FLAGS.num_filters
    self.keep_prob = FLAGS.keep_prob

    self.embedding = wenzheng.Embedding(vocab_size, 
                                        FLAGS.emb_dim, 
                                        FLAGS.word_embedding_file, 
                                        trainable=FLAGS.finetune_word_embedding)

    if FLAGS.use_label_emb or FLAGS.use_label_att:
      #assert not (FLAGS.use_label_emb and FLAGS.use_label_att)
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      if FLAGS.use_label_emb and FLAGS.use_label_att:
        assert self.label_emb_height == NUM_CLASSES * NUM_ATTRIBUTES
      self.label_embedding = melt.layers.Embedding(self.label_emb_height, FLAGS.emb_dim)
      if FLAGS.use_label_emb:
        #self.label_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
        self.label_dense = keras.layers.Dense(FLAGS.emb_dim, use_bias=False)
      if FLAGS.use_label_att:
        self.att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner)
        #self.att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=0.5, combiner=FLAGS.att_combiner)
        self.att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)
      if FLAGS.use_label_rnn:
        self.label_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)

    #self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)
    self.encode = wenzheng.Encoder(FLAGS.encoder_type)
    
    #self.multiplier = 2 if self.encode.bidirectional else 1

    # hier a bit worse
    self.hier_encode = melt.layers.HierEncode() if FLAGS.use_hier_encode else None

    if FLAGS.use_self_match:
      self.match_dot_attention = melt.layers.SelfAttnMatch(combiner=FLAGS.att_combiner, identity=True, diag=False, keep_prob=self.keep_prob)
      self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)
      #self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)
 
    # top-k best, max,att can benfit ensemble(better then max, worse then topk-3), topk,att now best with 2layers
    logging.info('encoder_output_method:', FLAGS.encoder_output_method)
    logging.info('topk:', FLAGS.top_k)
    self.pooling = melt.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        top_k=FLAGS.top_k,
                        att_activation=getattr(tf.nn, FLAGS.att_activation))
    #self.pooling = keras.layers.GlobalMaxPool1D()

    # mlp not help much!
    if FLAGS.mlp_ratio != 0:
      self.dropout = keras.layers.Dropout(0.3)
      if FLAGS.mlp_ratio < 0:
        # here activation hurt perf!
        #self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2, activation=tf.nn.relu)
        self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2)
      elif FLAGS.mlp_ratio <= 1:
        self.dense = melt.layers.DynamicDense(FLAGS.mlp_ratio)
      else:
        self.dense = kears.layers.Dense(int(FLAGS.mlp_ratio))
    else:
      self.dense = None

    self.num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
    self.logits = keras.layers.Dense(NUM_ATTRIBUTES * self.num_classes, activation=None)
    
  def call(self, input, training=False):
    x = input['content'] 

    c_mask = tf.cast(x, tf.bool)
    batch_size = melt.get_shape(x, 0)
    c_len = melt.length(x)

    if FLAGS.rnn_no_padding:
      logging.info('------------------no padding! train or eval')
      c_len = tf.ones([batch_size], dtype=x.dtype) * tf.cast(melt.get_shape(x, -1), x.dtype)

    #with tf.device('/cpu:0'):
    x = self.embedding(x)

    x = self.encode(x, c_len, training=training)
    #x = self.encode(x)

    # not help
    if self.hier_encode is not None:
      x = self.hier_encode(x, c_len)

    if FLAGS.use_label_att:
      label_emb = self.label_embedding(None)
      label_seq = tf.tile(tf.expand_dims(label_emb, 0), [batch_size, 1, 1])
      if FLAGS.use_label_rnn:
        label_seq = self.label_encode(label_seq, tf.ones([batch_size], tf.int32) * tf.cast(melt.get_shape(label_emb, 1), tf.int32))
      x = self.att_dot_attention(x, label_seq, mask=tf.ones([batch_size, self.label_emb_height], tf.bool), training=training)

      if not FLAGS.simple_label_att:
        x = self.att_encode(x, c_len, training=training)

    # pust self match at last
    if FLAGS.use_self_match:
       x = self.match_dot_attention(x, mask=c_mask, training=training) 
       x = self.match_encode(x, c_len, training=training) 

    x = self.pooling(x, c_len, calc_word_scores=self.debug)
    #x = self.pooling(x)

    # not help much
    if self.dense is not None:
      x = self.dense(x)
      x = self.dropout(x, training=training)

    if not FLAGS.use_label_emb:
      x = self.logits(x)
    else:
      x = self.label_dense(x)
      # TODO..
      x = melt.dot(x, self.label_embedding(None))

    # # No help match
    # if training and FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES * NUM_CLASSES:
    #   x = melt.adjust_lrs(x)

    x = tf.reshape(x, [batch_size, NUM_ATTRIBUTES, self.num_classes])
    
    return x

# currently Model as v1 is best v2 not improve, Model is like RNet
# NOTICE mainly use this one
class Model(melt.Model):
  def __init__(self):
    super(Model, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    logging.info('vocab_size:', vocab_size)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size if FLAGS.encoder_type != 'convnet' else FLAGS.num_filters
    self.keep_prob = FLAGS.keep_prob

    self.embedding = wenzheng.Embedding(vocab_size, 
                                        FLAGS.emb_dim, 
                                        FLAGS.word_embedding_file, 
                                        trainable=FLAGS.finetune_word_embedding)

    char_vocab_file = FLAGS.vocab.replace('vocab.txt', 'char_vocab.txt')
    if os.path.exists(char_vocab_file):
      FLAGS.use_char = True
      char_vocab = gezi.Vocabulary(char_vocab_file)
      logging.info('using char vocab:', char_vocab_file)
      self.char_embedding = wenzheng.Embedding(char_vocab.size(), 
                                               FLAGS.emb_dim, 
                                               FLAGS.word_embedding_file.replace('emb.npy', 'char_emb.npy') if FLAGS.word_embedding_file else None, 
                                               trainable=FLAGS.finetune_char_embedding)
    else:
      self.char_embedding = self.embedding

    #self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)
    self.encode = wenzheng.Encoder(FLAGS.encoder_type)

    if FLAGS.use_char:
      self.char_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob, cell=FLAGS.cell)
      self.char_pooling = melt.layers.Pooling(FLAGS.char_output_method)
      if FLAGS.char_combiner == 'sfu':
        self.char_sfu_combine = melt.layers.SemanticFusionCombine(keep_prob=self.keep_prob)

    if FLAGS.use_label_emb or FLAGS.use_label_att:
      #assert not (FLAGS.use_label_emb and FLAGS.use_label_att)
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      if FLAGS.use_label_emb and FLAGS.use_label_att:
        assert self.label_emb_height == NUM_CLASSES * NUM_ATTRIBUTES
      self.label_embedding = melt.layers.Embedding(self.label_emb_height, FLAGS.emb_dim)
      if FLAGS.use_label_emb:
        #self.label_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
        self.label_dense = keras.layers.Dense(FLAGS.emb_dim, use_bias=False)
      if FLAGS.use_label_att:
        self.att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner)
        #self.att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=0.5, combiner=FLAGS.att_combiner)
        self.att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob, cell=FLAGS.cell)
      if FLAGS.use_label_rnn:
        self.label_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob, cell=FLAGS.cell)

    
    #self.multiplier = 2 if self.encode.bidirectional else 1

    # hier a bit worse
    self.hier_encode = melt.layers.HierEncode() if FLAGS.use_hier_encode else None

    if FLAGS.use_self_match:
      self.match_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner)
      self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob, cell=FLAGS.cell)
      #self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)
 
    # top-k best, max,att can benfit ensemble(better then max, worse then topk-3), topk,att now best with 2layers
    logging.info('encoder_output_method:', FLAGS.encoder_output_method)
    logging.info('topk:', FLAGS.top_k)
    self.pooling = melt.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        top_k=FLAGS.top_k,
                        att_activation=getattr(tf.nn, FLAGS.att_activation))
    #self.pooling = keras.layers.GlobalMaxPool1D()

    # mlp not help much!
    if FLAGS.mlp_ratio != 0:
      self.dropout = keras.layers.Dropout(0.3)
      if FLAGS.mlp_ratio < 0:
        # here activation hurt perf!
        #self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2, activation=tf.nn.relu)
        self.dense = keras.layers.Dense(NUM_ATTRIBUTES * NUM_CLASSES * 2)
      elif FLAGS.mlp_ratio <= 1:
        self.dense = melt.layers.DynamicDense(FLAGS.mlp_ratio)
      else:
        self.dense = kears.layers.Dense(int(FLAGS.mlp_ratio))
    else:
      self.dense = None

    self.num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 1
    if FLAGS.loss_type == 'regression':
      self.num_classes = 1
    self.logits = keras.layers.Dense(NUM_ATTRIBUTES * self.num_classes, activation=None)
    
  def call(self, input, training=False):
    x = input['content'] 

    #print(input['source'])

    if FLAGS.content_limit and training:
      x = x[:,:FLAGS.content_limit]

    c_mask = tf.cast(x, tf.bool)
    batch_size = melt.get_shape(x, 0)
    c_len, max_c_len = melt.length2(x)

    if FLAGS.rnn_no_padding:
      logging.info('------------------no padding! train or eval')
      #c_len = tf.ones([batch_size], dtype=x.dtype) * tf.cast(melt.get_shape(x, -1), x.dtype)
      c_len = max_c_len

    #with tf.device('/cpu:0'):
    x = self.embedding(x)

    if FLAGS.use_char:
      cx = input['chars']
      cx = tf.reshape(cx, [batch_size * max_c_len, FLAGS.char_limit])
      chars_len = melt.length(cx)
      cx = self.char_embedding(cx)
      cx = self.char_encode(cx, chars_len, training=training)
      cx = self.char_pooling(cx, chars_len)
      cx = tf.reshape(cx, [batch_size, max_c_len, 2 * self.num_units])

      if FLAGS.char_combiner == 'concat':
        x = tf.concat([x, cx], axis=2)
      elif FLAGS.char_combiner == 'sfu':
        x = self.char_sfu_combine(x, cx, training=training)

    x = self.encode(x, c_len, training=training)
    #x = self.encode(x)

    # not help
    if self.hier_encode is not None:
      x = self.hier_encode(x, c_len)

    if FLAGS.use_label_att:
      label_emb = self.label_embedding(None)
      label_seq = tf.tile(tf.expand_dims(label_emb, 0), [batch_size, 1, 1])
      if FLAGS.use_label_rnn:
        label_seq = self.label_encode(label_seq, tf.ones([batch_size], tf.int32) * tf.cast(melt.get_shape(label_emb, 1), tf.int32))
      x = self.att_dot_attention(x, label_seq, mask=tf.ones([batch_size, self.label_emb_height], tf.bool), training=training)

      if not FLAGS.simple_label_att:
        x = self.att_encode(x, c_len, training=training)

    # put self match at last, selfmatch help a bit
    if FLAGS.use_self_match:
       x = self.match_dot_attention(x, x, mask=c_mask, training=training) 
       x = self.match_encode(x, c_len, training=training) 

    x = self.pooling(x, c_len, calc_word_scores=self.debug)
    #x = self.pooling(x)

    # not help much
    if self.dense is not None:
      x = self.dense(x)
      x = self.dropout(x, training=training)

    if not FLAGS.use_label_emb:
      x = self.logits(x)
    else:
      x = self.label_dense(x)
      # TODO..
      x = melt.dot(x, self.label_embedding(None))

    # # No help match
    # if training and FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES * NUM_CLASSES:
    #   x = melt.adjust_lrs(x)

    if FLAGS.loss_type == 'regression':
      x = tf.nn.sigmoid(x) * 10
    else:
      x = tf.reshape(x, [batch_size, NUM_ATTRIBUTES, self.num_classes])
      
    return x