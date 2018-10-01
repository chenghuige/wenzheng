#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   m_reader.py
#        \author   chenghuige  
#          \date   2018-09-28 15:16:16.146020
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

# same as V3 but not use share dropout and recurrent dropout
class MnemonicReaderV4(melt.Model):
  def __init__(self):
    super(MnemonicReaderV4, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    self.embedding = wenzheng.Embedding(vocab_size, 
                                        FLAGS.emb_dim, 
                                        FLAGS.word_embedding_file, 
                                        trainable=FLAGS.finetune_word_embedding,
                                        vocab2_size=FLAGS.unk_vocab_size,
                                        vocab2_trainable=FLAGS.finetune_unk_vocab)
    
    self.num_layers = FLAGS.num_layers
    
    if FLAGS.hop > 1:
      assert self.num_layers == 1 and FLAGS.att_combiner == 'sfu', 'mreader must set num layers to 1 so can iterative align if you set hop > 1, and use sfu as combiner'

    self.num_units = FLAGS.rnn_hidden_size
    self.keep_prob = FLAGS.keep_prob

    logging.info('num_layers:', self.num_layers)
    logging.info('num_unints:', self.num_units)
    logging.info('keep_prob:', self.keep_prob)

    self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob,
                                       recurrent_dropout=False)

    if FLAGS.use_qc_att or FLAGS.use_bidaf_att:
      assert not (FLAGS.use_qc_att and FLAGS.use_bidaf_att), 'use rnet or use bidaf? just choose one!'
      #Attention = melt.layers.DotAttention if FLAGS.use_qc_att else melt.layers.BiDAFAttention
      Attention = melt.layers.SeqAttnMatch if FLAGS.use_qc_att else melt.layers.BiDAFAttention
      # seems share att and match attention is fine a bit improve ? but just follow squad to use diffent dot attention 
      # NOTICE for eager mode ckpt save should not write as x = [None] * 3 can not save...
      self.att_dot_attentions = [] 
      self.att_encodes = [] 
      for _ in range(FLAGS.hop):
        #self.att_dot_attentions.append(Attention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner))
        #self.att_dot_attentions.append(Attention(keep_prob=self.keep_prob, combiner=FLAGS.att_combiner, identity=True))
        self.att_dot_attentions.append(Attention(combiner=FLAGS.att_combiner, identity=True))
        # TODO seems not work like layers.Dense... name in graph mode in eager mode will name as att_encode, match_encode 
        # in graph mode just cudnn_rnn, cudnn_rnn_1 so all ignore name=.. not like layers.Dense.. TODO
        # TODO seems in mreader do not use att_encode .. so check which is better use or not to use..
        self.att_encodes.append(melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob))
        #self.att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)

    if FLAGS.use_label_emb or FLAGS.use_label_att:
      assert not (FLAGS.use_label_emb and FLAGS.use_label_att)
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      self.label_embedding = melt.layers.Embedding(self.label_emb_height, FLAGS.emb_dim)
      if not FLAGS.use_label_att:
        # TODO not use activation ?
        #self.label_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
        self.label_dense = keras.layers.Dense(FLAGS.emb_dim)
      else:
        self.label_att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner)
        self.label_att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)
        #self.label_att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)
    
    if FLAGS.use_self_match:
      self.match_dot_attentions = []
      self.match_encodes = []
      for _ in range(FLAGS.hop):
        #self.match_dot_attentions.append(melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner))
        #self.match_dot_attentions.append(melt.layers.SelfAttnMatch(keep_prob=self.keep_prob, combiner=FLAGS.att_combiner, identity=True, diag=False))
        self.match_dot_attentions.append(melt.layers.SelfAttnMatch(combiner=FLAGS.att_combiner, identity=True, diag=False))
        self.match_encodes.append(melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob))
      #self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)

    if FLAGS.use_answer_emb:
      self.context_dense = keras.layers.Dense(FLAGS.emb_dim)
      self.answer_dense = keras.layers.Dense(FLAGS.emb_dim)
      # self.context_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
      # self.answer_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)

    if FLAGS.use_type_emb:
      type_emb_dim = 10
      num_types = 2
      self.type_embedding = melt.layers.Embedding(num_types, type_emb_dim)

    logging.info('encoder_output_method:', FLAGS.encoder_output_method)
    logging.info('topk:', FLAGS.top_k)
    self.pooling = melt.layers.Pooling(
                          FLAGS.encoder_output_method, 
                          top_k=FLAGS.top_k, 
                          att_activation=getattr(tf.nn, FLAGS.att_activation))

    self.logits = keras.layers.Dense(NUM_CLASSES)
    if FLAGS.split_type:
      self.logits2 = keras.layers.Dense(NUM_CLASSES)

  def call(self, input, training=False):
    q = input['query']
    c = input['passage']

    # reverse worse
    if FLAGS.cq_reverse:
      q, c = c, q

    #print(input['type'])
    # print('q', q)
    # print('c', c)

    q_len = melt.length(q)
    c_len = melt.length(c)
    q_mask = tf.cast(q, tf.bool)
    c_mask = tf.cast(c, tf.bool)

    q_emb = self.embedding(q)
    c_emb = self.embedding(c)
    
    x = c_emb
    batch_size = melt.get_shape(x, 0)

    if FLAGS.rnn_no_padding:
      logging.info('------------------no padding! train or eval')
      q_len = tf.ones([batch_size], dtype=q.dtype) * tf.cast(melt.get_shape(q, -1), q.dtype)
      c_len = tf.ones([batch_size], dtype=c.dtype) * tf.cast(melt.get_shape(c, -1), c.dtype)
      q_mask = tf.ones_like(q)
      c_mask = tf.ones_like(c)


    c = self.encode(c_emb, c_len, training=training)
    q = self.encode(q_emb, q_len, training=training)      

    # helps a lot using qc att, now bidaf att worse..
    # TODO... FIXME WRONG!  must use sfu as to iterative align gate will increase dim while sfu not
    x = c
    for i in range(FLAGS.hop):
      if not FLAGS.use_bidaf_att:
        x = self.att_dot_attentions[i](x, q, mask=q_mask, training=training)
      else:
        x = self.att_dot_attentions[i](x, q, c_mask, q_mask, training=training)
      if FLAGS.use_att_encode:
        x = self.att_encodes[i](x, c_len, training=training)
      #x = self.match_dot_attentions[i](x, x, mask=c_mask, training=training)
      x = self.match_dot_attentions[i](x, mask=c_mask, training=training)
      x = self.match_encodes[i](x, c_len, training=training)

    if FLAGS.mask_pooling:
      x = self.pooling(x, c_len, calc_word_scores=self.debug)
    else:
      x = self.pooling(x, None, calc_word_scores=self.debug)

    if FLAGS.use_type:
      x = tf.concat([x, tf.expand_dims(tf.to_float(input['type']), 1)], 1)
    
    if FLAGS.use_type_emb:
      x = tf.concat([x, self.type_embedding(input['type'])], 1)

    # might helps ensemble
    if FLAGS.use_answer_emb:
      x1 = x

      neg = input['candidate_neg']
      pos = input['candidate_pos']
      na = input['candidate_na']
      neg_len = melt.length(neg)
      pos_len = melt.length(pos)
      na_len = melt.length(na)
      neg_emb = self.embedding(neg)
      pos_emb = self.embedding(pos)
      na_emb = self.embedding(na)

      if FLAGS.share_dropout:
        neg = self.encode(neg_emb, neg_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
        pos = self.encode(pos_emb, pos_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
        na = self.encode(na_emb, na_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
      else:
        neg = self.encode(neg_emb, neg_len, training=training)
        pos = self.encode(pos_emb, pos_len, training=training)
        na = self.encode(na_emb, na_len, training=training)        

      neg = self.pooling(neg, neg_len)
      pos = self.pooling(pos, pos_len)
      na = self.pooling(na, na_len)

      answer = tf.stack([neg, pos, na], 1)

      # [batch_size, emb_dim]
      x = self.context_dense(x)
      # [batch_size, 3, emb_dim]
      answer = self.answer_dense(answer)
      x = tf.matmul(answer, tf.transpose(tf.expand_dims(x, 1), [0, 2, 1]))
      x = tf.reshape(x, [batch_size, NUM_CLASSES])

      x = tf.concat([x1, x], -1)

      #return x

    # not help
    if FLAGS.combine_query:
      q = self.pooling(q, q_len)
      x = tf.concat([x, q], -1)

    if not FLAGS.use_label_emb:
      # split logits by type is useful, especially for type1, and improve a lot with type1 only finetune
      if not FLAGS.split_type:
        x = self.logits(x)
      else:
        x1 = self.logits(x)
        x2 = self.logits2(x)
        mask = tf.expand_dims(tf.to_float(tf.equal(input['type'], 0)), 1)
        x = x1 * mask + x2 * (1 - mask)
    else:
      # use label emb seems not help ?
      x = self.label_dense(x)
      # TODO..
      x = melt.dot(x, self.label_embedding(None))

    return x
  
# this is follow hkust pytorch MnemonicReader with same attention implementation, this is V3
class MnemonicReader(melt.Model):
  def __init__(self):
    super(MnemonicReader, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    self.embedding = wenzheng.Embedding(vocab_size, 
                                        FLAGS.emb_dim, 
                                        FLAGS.word_embedding_file, 
                                        trainable=FLAGS.finetune_word_embedding,
                                        vocab2_size=FLAGS.unk_vocab_size,
                                        vocab2_trainable=FLAGS.finetune_unk_vocab)
    
    self.num_layers = FLAGS.num_layers
    
    if FLAGS.hop > 1:
      assert self.num_layers == 1 and FLAGS.att_combiner == 'sfu', 'mreader must set num layers to 1 so can iterative align if you set hop > 1, and use sfu as combiner'

    self.num_units = FLAGS.rnn_hidden_size
    self.keep_prob = FLAGS.keep_prob

    logging.info('num_layers:', self.num_layers)
    logging.info('num_unints:', self.num_units)
    logging.info('keep_prob:', self.keep_prob)

    self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)

    if FLAGS.use_qc_att or FLAGS.use_bidaf_att:
      assert not (FLAGS.use_qc_att and FLAGS.use_bidaf_att), 'use rnet or use bidaf? just choose one!'
      #Attention = melt.layers.DotAttention if FLAGS.use_qc_att else melt.layers.BiDAFAttention
      Attention = melt.layers.SeqAttnMatch if FLAGS.use_qc_att else melt.layers.BiDAFAttention
      # seems share att and match attention is fine a bit improve ? but just follow squad to use diffent dot attention 
      # NOTICE for eager mode ckpt save should not write as x = [None] * 3 can not save...
      self.att_dot_attentions = [] 
      self.att_encodes = [] 
      for _ in range(FLAGS.hop):
        #self.att_dot_attentions.append(Attention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner))
        #self.att_dot_attentions.append(Attention(keep_prob=self.keep_prob, combiner=FLAGS.att_combiner, identity=True))
        self.att_dot_attentions.append(Attention(combiner=FLAGS.att_combiner, identity=True))
        # TODO seems not work like layers.Dense... name in graph mode in eager mode will name as att_encode, match_encode 
        # in graph mode just cudnn_rnn, cudnn_rnn_1 so all ignore name=.. not like layers.Dense.. TODO
        # TODO seems in mreader do not use att_encode .. so check which is better use or not to use..
        self.att_encodes.append(melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob))
        #self.att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)

    if FLAGS.use_label_emb or FLAGS.use_label_att:
      assert not (FLAGS.use_label_emb and FLAGS.use_label_att)
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      self.label_embedding = melt.layers.Embedding(self.label_emb_height, FLAGS.emb_dim)
      if not FLAGS.use_label_att:
        # TODO not use activation ?
        #self.label_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
        self.label_dense = keras.layers.Dense(FLAGS.emb_dim)
      else:
        self.label_att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner)
        self.label_att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)
        #self.label_att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)
    
    if FLAGS.use_self_match:
      self.match_dot_attentions = []
      self.match_encodes = []
      for _ in range(FLAGS.hop):
        #self.match_dot_attentions.append(melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner))
        #self.match_dot_attentions.append(melt.layers.SelfAttnMatch(keep_prob=self.keep_prob, combiner=FLAGS.att_combiner, identity=True, diag=False))
        self.match_dot_attentions.append(melt.layers.SelfAttnMatch(combiner=FLAGS.att_combiner, identity=True, diag=False))
        self.match_encodes.append(melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob))
      #self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)

    if FLAGS.use_answer_emb:
      self.context_dense = keras.layers.Dense(FLAGS.emb_dim)
      self.answer_dense = keras.layers.Dense(FLAGS.emb_dim)
      # self.context_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
      # self.answer_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)

    if FLAGS.use_type_emb:
      type_emb_dim = 10
      num_types = 2
      self.type_embedding = melt.layers.Embedding(num_types, type_emb_dim)

    logging.info('encoder_output_method:', FLAGS.encoder_output_method)
    logging.info('topk:', FLAGS.top_k)
    self.pooling = melt.layers.Pooling(
                          FLAGS.encoder_output_method, 
                          top_k=FLAGS.top_k, 
                          att_activation=getattr(tf.nn, FLAGS.att_activation))

    self.logits = keras.layers.Dense(NUM_CLASSES)
    if FLAGS.split_type:
      self.logits2 = keras.layers.Dense(NUM_CLASSES)

  def call(self, input, training=False):
    q = input['query']
    c = input['passage']

    # reverse worse
    if FLAGS.cq_reverse:
      q, c = c, q

    #print(input['type'])
    # print('q', q)
    # print('c', c)

    q_len = melt.length(q)
    c_len = melt.length(c)
    q_mask = tf.cast(q, tf.bool)
    c_mask = tf.cast(c, tf.bool)

    q_emb = self.embedding(q)
    c_emb = self.embedding(c)
    
    x = c_emb
    batch_size = melt.get_shape(x, 0)

    if FLAGS.rnn_no_padding:
      logging.info('------------------no padding! train or eval')
      q_len = tf.ones([batch_size], dtype=q.dtype) * tf.cast(melt.get_shape(q, -1), q.dtype)
      c_len = tf.ones([batch_size], dtype=c.dtype) * tf.cast(melt.get_shape(c, -1), c.dtype)
      q_mask = tf.ones_like(q)
      c_mask = tf.ones_like(c)

    if FLAGS.share_dropout:
      num_units = [melt.get_shape(x, -1) if layer == 0 else 2 * self.num_units for layer in range(self.num_layers)]
      mask_fws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
      mask_bws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
      
      # NOTICE query and passage share same drop out, so same word still has same embedding vector after dropout in query and passage
      c = self.encode(c_emb, c_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
      q = self.encode(q_emb, q_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
    else:
      c = self.encode(c_emb, c_len, training=training)
      q = self.encode(q_emb, q_len, training=training)      

    # helps a lot using qc att, now bidaf att worse..
    # TODO... FIXME WRONG!  must use sfu as to iterative align gate will increase dim while sfu not
    x = c
    for i in range(FLAGS.hop):
      if not FLAGS.use_bidaf_att:
        x = self.att_dot_attentions[i](x, q, mask=q_mask, training=training)
      else:
        x = self.att_dot_attentions[i](x, q, c_mask, q_mask, training=training)
      if FLAGS.use_att_encode:
        x = self.att_encodes[i](x, c_len, training=training)
      #x = self.match_dot_attentions[i](x, x, mask=c_mask, training=training)
      x = self.match_dot_attentions[i](x, mask=c_mask, training=training)
      x = self.match_encodes[i](x, c_len, training=training)

    if FLAGS.mask_pooling:
      x = self.pooling(x, c_len, calc_word_scores=self.debug)
    else:
      x = self.pooling(x, None, calc_word_scores=self.debug)

    if FLAGS.use_type:
      x = tf.concat([x, tf.expand_dims(tf.to_float(input['type']), 1)], 1)
    
    if FLAGS.use_type_emb:
      x = tf.concat([x, self.type_embedding(input['type'])], 1)

    # might helps ensemble
    if FLAGS.use_answer_emb:
      x1 = x

      neg = input['candidate_neg']
      pos = input['candidate_pos']
      na = input['candidate_na']
      neg_len = melt.length(neg)
      pos_len = melt.length(pos)
      na_len = melt.length(na)
      neg_emb = self.embedding(neg)
      pos_emb = self.embedding(pos)
      na_emb = self.embedding(na)

      if FLAGS.share_dropout:
        neg = self.encode(neg_emb, neg_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
        pos = self.encode(pos_emb, pos_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
        na = self.encode(na_emb, na_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
      else:
        neg = self.encode(neg_emb, neg_len, training=training)
        pos = self.encode(pos_emb, pos_len, training=training)
        na = self.encode(na_emb, na_len, training=training)        

      neg = self.pooling(neg, neg_len)
      pos = self.pooling(pos, pos_len)
      na = self.pooling(na, na_len)

      answer = tf.stack([neg, pos, na], 1)

      # [batch_size, emb_dim]
      x = self.context_dense(x)
      # [batch_size, 3, emb_dim]
      answer = self.answer_dense(answer)
      x = tf.matmul(answer, tf.transpose(tf.expand_dims(x, 1), [0, 2, 1]))
      x = tf.reshape(x, [batch_size, NUM_CLASSES])

      x = tf.concat([x1, x], -1)

      #return x

    # not help
    if FLAGS.combine_query:
      q = self.pooling(q, q_len)
      x = tf.concat([x, q], -1)

    if not FLAGS.use_label_emb:
      # split logits by type is useful, especially for type1, and improve a lot with type1 only finetune
      if not FLAGS.split_type:
        x = self.logits(x)
      else:
        x1 = self.logits(x)
        x2 = self.logits2(x)
        mask = tf.expand_dims(tf.to_float(tf.equal(input['type'], 0)), 1)
        x = x1 * mask + x2 * (1 - mask)
    else:
      # use label emb seems not help ?
      x = self.label_dense(x)
      # TODO..
      x = melt.dot(x, self.label_embedding(None))

    return x

# V2 is follow hkust pytorch MnemonicReader but more dropout on attention(like hkust tensorflow rnet)
class MnemonicReaderV2(melt.Model):
  def __init__(self):
    super(MnemonicReaderV2, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    self.embedding = wenzheng.Embedding(vocab_size, 
                                        FLAGS.emb_dim, 
                                        FLAGS.word_embedding_file, 
                                        trainable=FLAGS.finetune_word_embedding,
                                        vocab2_size=FLAGS.unk_vocab_size,
                                        vocab2_trainable=FLAGS.finetune_unk_vocab)
    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.keep_prob = FLAGS.keep_prob

    logging.info('num_layers:', self.num_layers)
    logging.info('num_unints:', self.num_units)
    logging.info('keep_prob:', self.keep_prob)

    self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)

    if FLAGS.use_qc_att or FLAGS.use_bidaf_att:
      assert not (FLAGS.use_qc_att and FLAGS.use_bidaf_att), 'use rnet or use bidaf? just choose one!'
      #Attention = melt.layers.DotAttention if FLAGS.use_qc_att else melt.layers.BiDAFAttention
      Attention = melt.layers.SeqAttnMatch if FLAGS.use_qc_att else melt.layers.BiDAFAttention
      # seems share att and match attention is fine a bit improve ? but just follow squad to use diffent dot attention 
      # NOTICE for eager mode ckpt save should not write as x = [None] * 3 can not save...
      self.att_dot_attentions = [] 
      self.att_encodes = [] 
      for _ in range(FLAGS.hop):
        #self.att_dot_attentions.append(Attention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner))
        self.att_dot_attentions.append(Attention(keep_prob=self.keep_prob, combiner=FLAGS.att_combiner, identity=True))
        #self.att_dot_attentions.append(Attention(combiner=FLAGS.att_combiner, identity=True))
        # TODO seems not work like layers.Dense... name in graph mode in eager mode will name as att_encode, match_encode 
        # in graph mode just cudnn_rnn, cudnn_rnn_1 so all ignore name=.. not like layers.Dense.. TODO
        # TODO seems in mreader do not use att_encode .. so check which is better use or not to use..
        self.att_encodes.append(melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob))
        #self.att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)

    if FLAGS.use_label_emb or FLAGS.use_label_att:
      assert not (FLAGS.use_label_emb and FLAGS.use_label_att)
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      self.label_embedding = melt.layers.Embedding(self.label_emb_height, FLAGS.emb_dim)
      if not FLAGS.use_label_att:
        # TODO not use activation ?
        #self.label_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
        self.label_dense = keras.layers.Dense(FLAGS.emb_dim)
      else:
        self.label_att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner)
        self.label_att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)
        #self.label_att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)
    
    if FLAGS.use_self_match:
      self.match_dot_attentions = []
      self.match_encodes = []
      for _ in range(FLAGS.hop):
        #self.match_dot_attentions.append(melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner))
        self.match_dot_attentions.append(melt.layers.SelfAttnMatch(keep_prob=self.keep_prob, combiner=FLAGS.att_combiner, identity=True, diag=False))
        #self.match_dot_attentions.append(melt.layers.SelfAttnMatch(combiner=FLAGS.att_combiner, identity=True, diag=False))
        self.match_encodes.append(melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob))
      #self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)

    if FLAGS.use_answer_emb:
      self.context_dense = keras.layers.Dense(FLAGS.emb_dim)
      self.answer_dense = keras.layers.Dense(FLAGS.emb_dim)
      # self.context_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
      # self.answer_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)

    logging.info('encoder_output_method:', FLAGS.encoder_output_method)
    logging.info('topk:', FLAGS.top_k)
    self.pooling = melt.layers.Pooling(
                          FLAGS.encoder_output_method, 
                          top_k=FLAGS.top_k, 
                          att_activation=getattr(tf.nn, FLAGS.att_activation))

    self.logits = keras.layers.Dense(NUM_CLASSES)
    if FLAGS.split_type:
      self.logits2 = keras.layers.Dense(NUM_CLASSES)

  def call(self, input, training=False):
    q = input['query']
    c = input['passage']

    # reverse worse
    if FLAGS.cq_reverse:
      q, c = c, q

    #print(input['type'])
    # print('q', q)
    # print('c', c)

    q_len = melt.length(q)
    c_len = melt.length(c)
    q_mask = tf.cast(q, tf.bool)
    c_mask = tf.cast(c, tf.bool)
    q_emb = self.embedding(q)
    c_emb = self.embedding(c)
    
    x = c_emb
    batch_size = melt.get_shape(x, 0)

    if FLAGS.share_dropout:
      num_units = [melt.get_shape(x, -1) if layer == 0 else 2 * self.num_units for layer in range(self.num_layers)]
      mask_fws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
      mask_bws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
      
      # NOTICE query and passage share same drop out, so same word still has same embedding vector after dropout in query and passage
      c = self.encode(c_emb, c_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
      q = self.encode(q_emb, q_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
    else:
      c = self.encode(c_emb, c_len, training=training)
      q = self.encode(q_emb, q_len, training=training)      

    # helps a lot using qc att, now bidaf att worse..
    for i in range(FLAGS.hop):
      if not FLAGS.use_bidaf_att:
        x = self.att_dot_attentions[i](c, q, mask=q_mask, training=training)
      else:
        x = self.att_dot_attentions[i](c, q, c_mask, q_mask, training=training)
      if FLAGS.use_att_encode:
        x = self.att_encodes[i](x, c_len, training=training)
      #x = self.match_dot_attentions[i](x, x, mask=c_mask, training=training)
      x = self.match_dot_attentions[i](x, mask=c_mask, training=training)
      x = self.match_encodes[i](x, c_len, training=training)

    x = self.pooling(x, c_len, calc_word_scores=self.debug)

    if FLAGS.use_type:
      x = tf.concat([x, tf.expand_dims(tf.to_float(input['type']), 1)], 1)

    # might helps ensemble
    if FLAGS.use_answer_emb:
      x1 = x

      neg = input['candidate_neg']
      pos = input['candidate_pos']
      na = input['candidate_na']
      neg_len = melt.length(neg)
      pos_len = melt.length(pos)
      na_len = melt.length(na)
      neg_emb = self.embedding(neg)
      pos_emb = self.embedding(pos)
      na_emb = self.embedding(na)

      if FLAGS.share_dropout:
        neg = self.encode(neg_emb, neg_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
        pos = self.encode(pos_emb, pos_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
        na = self.encode(na_emb, na_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
      else:
        neg = self.encode(neg_emb, neg_len, training=training)
        pos = self.encode(pos_emb, pos_len, training=training)
        na = self.encode(na_emb, na_len, training=training)        

      neg = self.pooling(neg, neg_len)
      pos = self.pooling(pos, pos_len)
      na = self.pooling(na, na_len)

      answer = tf.stack([neg, pos, na], 1)

      # [batch_size, emb_dim]
      x = self.context_dense(x)
      # [batch_size, 3, emb_dim]
      answer = self.answer_dense(answer)
      x = tf.matmul(answer, tf.transpose(tf.expand_dims(x, 1), [0, 2, 1]))
      x = tf.reshape(x, [batch_size, NUM_CLASSES])

      x = tf.concat([x1, x], -1)

      #return x

    # not help
    if FLAGS.combine_query:
      q = self.pooling(q, q_len)
      x = tf.concat([x, q], -1)

    if not FLAGS.use_label_emb:
      # split logits by type is useful, especially for type1, and improve a lot with type1 only finetune
      if not FLAGS.split_type:
        x = self.logits(x)
      else:
        x1 = self.logits(x)
        x2 = self.logits2(x)
        mask = tf.expand_dims(tf.to_float(tf.equal(input['type'], 0)), 1)
        x = x1 * mask + x2 * (1 - mask)
    else:
      # use label emb seems not help ?
      x = self.label_dense(x)
      # TODO..
      x = melt.dot(x, self.label_embedding(None))

    return x

# V1 is just like hkust rnet with interactive attention
class MnemonicReaderV1(melt.Model):
  def __init__(self):
    super(MnemonicReaderV1, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    self.embedding = wenzheng.Embedding(vocab_size, 
                                        FLAGS.emb_dim, 
                                        FLAGS.word_embedding_file, 
                                        trainable=FLAGS.finetune_word_embedding,
                                        vocab2_size=FLAGS.unk_vocab_size,
                                        vocab2_trainable=FLAGS.finetune_unk_vocab)
    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.keep_prob = FLAGS.keep_prob

    logging.info('num_layers:', self.num_layers)
    logging.info('num_unints:', self.num_units)
    logging.info('keep_prob:', self.keep_prob)

    self.encode = melt.layers.CudnnRnn(num_layers=self.num_layers, num_units=self.num_units, keep_prob=self.keep_prob)

    if FLAGS.use_qc_att or FLAGS.use_bidaf_att:
      assert not (FLAGS.use_qc_att and FLAGS.use_bidaf_att), 'use rnet or use bidaf? just choose one!'
      #Attention = melt.layers.DotAttention if FLAGS.use_qc_att else melt.layers.BiDAFAttention
      Attention = melt.layers.SeqAttnMatch if FLAGS.use_qc_att else melt.layers.BiDAFAttention
      # seems share att and match attention is fine a bit improve ? but just follow squad to use diffent dot attention 
      # NOTICE for eager mode ckpt save should not write as x = [None] * 3 can not save...
      self.att_dot_attentions = [] 
      self.att_encodes = [] 
      for _ in range(FLAGS.hop):
        self.att_dot_attentions.append(Attention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner))
        #self.att_dot_attentions.append(Attention(keep_prob=self.keep_prob, combiner=FLAGS.att_combiner, identity=True))
        #self.att_dot_attentions.append(Attention(combiner=FLAGS.att_combiner, identity=True))
        # TODO seems not work like layers.Dense... name in graph mode in eager mode will name as att_encode, match_encode 
        # in graph mode just cudnn_rnn, cudnn_rnn_1 so all ignore name=.. not like layers.Dense.. TODO
        # TODO seems in mreader do not use att_encode .. so check which is better use or not to use..
        self.att_encodes.append(melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob))
        #self.att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)

    if FLAGS.use_label_emb or FLAGS.use_label_att:
      assert not (FLAGS.use_label_emb and FLAGS.use_label_att)
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      self.label_embedding = melt.layers.Embedding(self.label_emb_height, FLAGS.emb_dim)
      if not FLAGS.use_label_att:
        # TODO not use activation ?
        #self.label_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
        self.label_dense = keras.layers.Dense(FLAGS.emb_dim)
      else:
        self.label_att_dot_attention = melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner)
        self.label_att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob)
        #self.label_att_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)
    
    if FLAGS.use_self_match:
      self.match_dot_attentions = []
      self.match_encodes = []
      for _ in range(FLAGS.hop):
        self.match_dot_attentions.append(melt.layers.DotAttention(hidden=self.num_units, keep_prob=self.keep_prob, combiner=FLAGS.att_combiner))
        #self.match_dot_attentions.append(melt.layers.SelfAttnMatch(keep_prob=self.keep_prob, combiner=FLAGS.att_combiner, identity=True, diag=False))
        #self.match_dot_attentions.append(melt.layers.SelfAttnMatch(combiner=FLAGS.att_combiner, identity=True, diag=False))
        self.match_encodes.append(melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=self.keep_prob))
      #self.match_encode = melt.layers.CudnnRnn(num_layers=1, num_units=self.num_units, keep_prob=0.5)

    if FLAGS.use_answer_emb:
      self.context_dense = keras.layers.Dense(FLAGS.emb_dim)
      self.answer_dense = keras.layers.Dense(FLAGS.emb_dim)
      # self.context_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)
      # self.answer_dense = keras.layers.Dense(FLAGS.emb_dim, activation=tf.nn.relu)

    logging.info('encoder_output_method:', FLAGS.encoder_output_method)
    logging.info('topk:', FLAGS.top_k)
    self.pooling = melt.layers.Pooling(
                          FLAGS.encoder_output_method, 
                          top_k=FLAGS.top_k, 
                          att_activation=getattr(tf.nn, FLAGS.att_activation))

    self.logits = keras.layers.Dense(NUM_CLASSES)
    if FLAGS.split_type:
      self.logits2 = keras.layers.Dense(NUM_CLASSES)

  def call(self, input, training=False):
    q = input['query']
    c = input['passage']

    # reverse worse
    if FLAGS.cq_reverse:
      q, c = c, q

    #print(input['type'])
    # print('q', q)
    # print('c', c)

    q_len = melt.length(q)
    c_len = melt.length(c)
    q_mask = tf.cast(q, tf.bool)
    c_mask = tf.cast(c, tf.bool)
    q_emb = self.embedding(q)
    c_emb = self.embedding(c)
    
    x = c_emb
    batch_size = melt.get_shape(x, 0)

    if FLAGS.share_dropout:
      num_units = [melt.get_shape(x, -1) if layer == 0 else 2 * self.num_units for layer in range(self.num_layers)]
      mask_fws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
      mask_bws = [melt.dropout(tf.ones([batch_size, 1, num_units[layer]], dtype=tf.float32), keep_prob=self.keep_prob, training=training, mode=None) for layer in range(self.num_layers)]
      
      # NOTICE query and passage share same drop out, so same word still has same embedding vector after dropout in query and passage
      c = self.encode(c_emb, c_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
      q = self.encode(q_emb, q_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
    else:
      c = self.encode(c_emb, c_len, training=training)
      q = self.encode(q_emb, q_len, training=training)      

    # helps a lot using qc att, now bidaf att worse..
    for i in range(FLAGS.hop):
      if not FLAGS.use_bidaf_att:
        x = self.att_dot_attentions[i](c, q, mask=q_mask, training=training)
      else:
        x = self.att_dot_attentions[i](c, q, c_mask, q_mask, training=training)
      if FLAGS.use_att_encode:
        x = self.att_encodes[i](x, c_len, training=training)
      x = self.match_dot_attentions[i](x, x, mask=c_mask, training=training)
      #x = self.match_dot_attentions[i](x, mask=c_mask, training=training)
      x = self.match_encodes[i](x, c_len, training=training)

    x = self.pooling(x, c_len, calc_word_scores=self.debug)

    if FLAGS.use_type:
      x = tf.concat([x, tf.expand_dims(tf.to_float(input['type']), 1)], 1)

    # might helps ensemble
    if FLAGS.use_answer_emb:
      x1 = x

      neg = input['candidate_neg']
      pos = input['candidate_pos']
      na = input['candidate_na']
      neg_len = melt.length(neg)
      pos_len = melt.length(pos)
      na_len = melt.length(na)
      neg_emb = self.embedding(neg)
      pos_emb = self.embedding(pos)
      na_emb = self.embedding(na)

      if FLAGS.share_dropout:
        neg = self.encode(neg_emb, neg_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
        pos = self.encode(pos_emb, pos_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
        na = self.encode(na_emb, na_len, mask_fws=mask_fws, mask_bws=mask_bws, training=training)
      else:
        neg = self.encode(neg_emb, neg_len, training=training)
        pos = self.encode(pos_emb, pos_len, training=training)
        na = self.encode(na_emb, na_len, training=training)        

      neg = self.pooling(neg, neg_len)
      pos = self.pooling(pos, pos_len)
      na = self.pooling(na, na_len)

      answer = tf.stack([neg, pos, na], 1)

      # [batch_size, emb_dim]
      x = self.context_dense(x)
      # [batch_size, 3, emb_dim]
      answer = self.answer_dense(answer)
      x = tf.matmul(answer, tf.transpose(tf.expand_dims(x, 1), [0, 2, 1]))
      x = tf.reshape(x, [batch_size, NUM_CLASSES])

      x = tf.concat([x1, x], -1)

      #return x

    # not help
    if FLAGS.combine_query:
      q = self.pooling(q, q_len)
      x = tf.concat([x, q], -1)

    if not FLAGS.use_label_emb:
      # split logits by type is useful, especially for type1, and improve a lot with type1 only finetune
      if not FLAGS.split_type:
        x = self.logits(x)
      else:
        x1 = self.logits(x)
        x2 = self.logits2(x)
        mask = tf.expand_dims(tf.to_float(tf.equal(input['type'], 0)), 1)
        x = x1 * mask + x2 * (1 - mask)
    else:
      # use label emb seems not help ?
      x = self.label_dense(x)
      # TODO..
      x = melt.dot(x, self.label_embedding(None))

    return x