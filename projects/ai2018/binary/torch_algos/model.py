#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2018-09-30 10:25:11.024133
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import torch
from torch import nn
from torch.nn import functional as F

import wenzheng
from wenzheng.utils import vocabulary

import melt
logging = melt.logging

import numpy as np
import lele
layers = lele.layers
import gezi
import os

class ModelBase(nn.Module):
  def __init__(self, embedding=None, lm_model=False):
    super(ModelBase, self).__init__()
    
    self.num_units = FLAGS.rnn_hidden_size
    self.dropout_rate = 1 - FLAGS.keep_prob

    config = {
      'word': {
        'vocab': FLAGS.vocab,
        'num_layers': FLAGS.num_layers,
        'hidden_size': FLAGS.rnn_hidden_size,
        'emb_dim': FLAGS.emb_dim,
        'embedding_file': FLAGS.word_embedding_file,
        'trainable': FLAGS.finetune_word_embedding,
        'num_finetune': FLAGS.num_finetune_words,
      },
      'char': {
        'limit': FLAGS.char_limit,
        'use_char_emb': FLAGS.use_char_emb,
        'emb_dim': FLAGS.char_emb_dim,
        'trainable': FLAGS.finetune_char_embedding,
        'hidden_size': FLAGS.rnn_hidden_size,
        'output_method': FLAGS.char_output_method,
        'combiner': FLAGS.char_combiner,
        'padding': FLAGS.char_padding,
        'num_finetune': FLAGS.num_finetune_chars,
      },
      'pos': {
        'emb_dim': FLAGS.tag_emb_dim 
      },
      'ner': {
        'emb_dim': FLAGS.tag_emb_dim 
      },
      'encoder': FLAGS.encoder_type,
      'dropout_rate': 1 - FLAGS.keep_prob,
      'recurrent_dropout': FLAGS.recurrent_dropout,
      'cell': FLAGS.cell,
      'rnn_padding': FLAGS.rnn_padding,
      'rnn_no_padding': FLAGS.rnn_no_padding,
      'concat_layers': FLAGS.concat_layers,
    }

    self.encode = wenzheng.pyt.TextEncoder(config, 
                                         embedding,
                                         use_char=FLAGS.use_char,
                                         use_char_emb=FLAGS.use_char_emb,
                                         use_pos=FLAGS.use_pos,
                                         use_ner=FLAGS.use_ner,
                                         lm_model=lm_model)

    self.lm_model = self.encode.lm_model

    if not self.lm_model:
      doc_hidden_size = self.encode.output_size
      if FLAGS.share_pooling:
        self.pooling = lele.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        input_size=doc_hidden_size,
                        top_k=FLAGS.top_k, 
                        att_activation=getattr(F, FLAGS.att_activation)) 
      else:
        self.pooling = lele.layers.Poolings(
                        FLAGS.encoder_output_method, 
                        input_size=doc_hidden_size,
                        num_poolings=NUM_ATTRIBUTES,
                        top_k=FLAGS.top_k, 
                        att_activation=getattr(F, FLAGS.att_activation))         

      # input dim not as convinient as tf..
      pre_logits_dim = self.pooling.output_size
      self.logits = nn.Linear(pre_logits_dim, 1)

  def unk_aug(self, x, x_mask=None):
    """
    randomly make 10% words as unk
    TODO this works, but should this be rmoved and put it to Dataset so can share for both pyt and tf
    """
    if not self.training or not FLAGS.unk_aug or melt.epoch() < FLAGS.unk_aug_start_epoch:
      return x 

    if x_mask is None:
      x_mask = x > 0
    x_mask = x_mask.long()

    ratio = np.random.uniform(0, FLAGS.unk_aug_max_ratio)
    mask = torch.cuda.FloatTensor(x.size(0), x.size(1)).uniform_() > ratio
    mask = mask.long()
    rmask = FLAGS.unk_id * (1 - mask)

    x = (x * mask + rmask) * x_mask
    return x

class BiLanguageModel(ModelBase):
  def __init__(self, embedding=None):
    super(BiLanguageModel, self).__init__(embedding, lm_model=True)
    
# Model is like RNet! if you use label att and self match
# if not just simple gru model
class RNet(ModelBase):
  def __init__(self, embedding=None):
    super(RNet, self).__init__(embedding)

    Rnn = lele.layers.StackedBRNN     
    doc_hidden_size = self.encode.output_size

    if FLAGS.use_label_att:
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      self.label_embedding = nn.Embedding(self.label_emb_height, FLAGS.emb_dim)

      self.att_dot_attentions = nn.ModuleList()
      self.att_encodes = nn.ModuleList()
      for i in range(FLAGS.label_hop):
        self.att_dot_attentions.append(lele.layers.DotAttention(input_size=doc_hidden_size, 
                                                                input_size2=FLAGS.emb_dim,
                                                                hidden=self.num_units, 
                                                                dropout_rate=self.dropout_rate, 
                                                                combiner=FLAGS.att_combiner))
        self.att_encodes.append(Rnn(
              input_size=self.att_dot_attentions[-1].output_size,
              hidden_size=self.num_units,
              num_layers=1,
              dropout_rate=self.dropout_rate,
              dropout_output=False,
              concat_layers=False,
              rnn_type=FLAGS.cell,
              padding=FLAGS.rnn_padding,
          ))

    if FLAGS.use_self_match:
      self.match_dot_attention = lele.layers.DotAttention(input_size=doc_hidden_size, 
                                                          input_size2=doc_hidden_size, 
                                                          hidden=self.num_units, 
                                                          dropout_rate=self.dropout_rate, 
                                                          combiner=FLAGS.att_combiner)
      self.match_encode = Rnn(
            input_size=self.match_dot_attention.output_size,
            hidden_size=self.num_units,
            num_layers=1,
            dropout_rate=self.dropout_rate,
            dropout_output=False,
            concat_layers=False,
            rnn_type=FLAGS.cell,
            padding=FLAGS.rnn_padding,
        )    

  def forward(self, input, training=False):
    x = input['content'] 
    #print(x.shape)
    x_mask = x.eq(0)
    batch_size = x.size(0)

    x = self.unk_aug(x, x_mask)

    if FLAGS.rnn_no_padding:
      x_mask = torch.zeros_like(x, dtype=torch.uint8)

    x = self.encode(input, x_mask, training=training)

    if FLAGS.use_label_att:
      label_emb = self.label_embedding.weight
      label_seq = lele.tile(label_emb.unsqueeze(0), 0, batch_size)
      # TODO label rnn 
      for i in range(FLAGS.label_hop):
        x = self.att_dot_attentions[i](x, label_seq, torch.zeros(batch_size, self.label_emb_height).byte().cuda())
        x = self.att_encodes[i](x, x_mask)

    if FLAGS.use_self_match:
       x = self.match_dot_attention(x, x, x_mask) 
       x = self.match_encode(x, x_mask) 

    x = self.pooling(x, x_mask)
    
    x = self.logits(x)  

    return x
  
# MReader with hop==1 can be viewed as rnet also
# NOTICE mainly use this one
# TODO word and char encoder can merge to one Encoder handle inputs with word and char
class MReader(ModelBase):
  def __init__(self, embedding=None):
    super(MReader, self).__init__(embedding)

    Rnn = lele.layers.StackedBRNN 
    doc_hidden_size = self.encode.output_size

    if FLAGS.use_label_att:
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      self.label_embedding = nn.Embedding(self.label_emb_height, FLAGS.emb_dim)
      # here linear better or another rnn is better ?
      if not FLAGS.use_label_rnn:
        self.label_forward = nn.Linear(FLAGS.emb_dim, doc_hidden_size)
      else:
        self.label_forward = Rnn(
              input_size=emb_dim,
              hidden_size=self.num_units,
              num_layers=1,
              dropout_rate=self.dropout_rate,
              dropout_output=False,
              concat_layers=False,
              rnn_type=FLAGS.cell,
              padding=FLAGS.rnn_padding,
          )    

    # Interactive aligning, self aligning and aggregating
    self.interactive_aligners = nn.ModuleList()
    self.interactive_SFUs = nn.ModuleList()
    self.self_aligners = nn.ModuleList()
    self.self_SFUs = nn.ModuleList()
    self.aggregate_rnns = nn.ModuleList()
    
    for i in range(FLAGS.hop):
      # interactive aligner
      if FLAGS.use_label_att:
        self.interactive_aligners.append(layers.SeqAttnMatch(doc_hidden_size, identity=True))
        self.interactive_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
      # self aligner
      if FLAGS.use_self_match:
        self.self_aligners.append(layers.SelfAttnMatch(doc_hidden_size, identity=True, diag=False))
        self.self_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
        # aggregating
        self.aggregate_rnns.append(
            Rnn(
                input_size=doc_hidden_size,
                hidden_size=self.num_units,
                num_layers=1,
                dropout_rate=self.dropout_rate,
                dropout_output=False,
                concat_layers=False,
                rnn_type=FLAGS.cell,
                padding=FLAGS.rnn_padding,
            )
        )

  def forward(self, input, training=False):
    #print('------------', input['source'])
    #print(input['id'])
    x = input['content']
    #print(x) 
    #print(x.shape)
    x_mask = x.eq(0)
    batch_size = x.size(0)
    max_c_len = x.size(1)

    x = self.unk_aug(x, x_mask)

    if FLAGS.rnn_no_padding:
      x_mask = torch.zeros_like(x, dtype=torch.uint8)

    x = self.encode(input, x_mask, training=training)

    if FLAGS.use_label_att:
      label_emb = self.label_embedding.weight
      label_seq = lele.tile(label_emb.unsqueeze(0), 0, batch_size)
      x2_mask = torch.zeros(batch_size, self.label_emb_height).byte().cuda()
      if not FLAGS.use_label_rnn:
        label_seq = self.label_forward(label_seq)
      else:
        label_seq = self.label_forward(label_seq, x2_mask)
      # Align and aggregate
      c_check = x
      q = label_seq
    else:
      c_check = x
    
    #print(c_check.shape, q.shape, x2_mask.shape)
    for i in range(FLAGS.hop):
      if FLAGS.use_label_att:
        q_tilde = self.interactive_aligners[i].forward(c_check, q, x2_mask)
        c_bar = self.interactive_SFUs[i].forward(c_check, torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))
      else:
        c_bar = c_check
      if FLAGS.use_self_match:
        c_tilde = self.self_aligners[i].forward(c_bar, x_mask)
        c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
        c_check = self.aggregate_rnns[i].forward(c_hat, x_mask)
    
    x = c_check

    x = self.pooling(x, x_mask)

    x = self.logits(x)  

    return x
  
# class Model(ModelBase):
#   def forward(self, input, training=False):
#     x = input['content'] 
#     x_mask = x.eq(0)
#     batch_size = x.size(0)
#     x = self.encode(input, x_mask, training=training)
#     x = self.pooling(x, x_mask)
#     x = self.logits(x)  
#     x = x.view([-1, NUM_ATTRIBUTES, self.num_classes])
#     return x

