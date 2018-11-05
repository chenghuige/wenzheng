#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2018-09-28 10:09:41.585876
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
# from torch_algos.rnet import Rnet
# from torch_algos.m_reader import *
# from torch_algos.m_reader import MnemonicReader 
# MReader = MnemonicReader 
# # baseline
# from torch_algos.baseline.baseline import *

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import torch
import torch.nn as nn
import torch.nn.functional as F
import lele
layers = lele.layers
from torch.autograd import Variable

import wenzheng
from wenzheng.utils import vocabulary
import wenzheng.utils.input_flags

from algos.config import NUM_CLASSES

import numpy as np

import melt
logging = melt.logging

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
    self.config = config

def get_mask(x):
  if FLAGS.rnn_no_padding:
    x_mask = torch.zeros_like(x, dtype=torch.uint8)
  else:
    x_mask = x.eq(0)
  return x_mask

class MReader(ModelBase):
  def __init__(self, embedding=None):
    super(MReader, self).__init__(embedding)

    Rnn = lele.layers.StackedBRNN 
    doc_hidden_size = self.encode.output_size        
    
    # Interactive aligning, self aligning and aggregating
    self.interactive_aligners = nn.ModuleList()
    self.interactive_SFUs = nn.ModuleList()
    self.self_aligners = nn.ModuleList()
    self.self_SFUs = nn.ModuleList()
    self.aggregate_rnns = nn.ModuleList()
    
    for i in range(FLAGS.hop):
      # interactive aligner
      self.interactive_aligners.append(layers.SeqAttnMatch(doc_hidden_size, identity=True))
      self.interactive_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
      # self aligner
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

    self.pooling = lele.layers.Pooling(
                    FLAGS.encoder_output_method, 
                    input_size=doc_hidden_size,
                    top_k=FLAGS.top_k, 
                    att_activation=getattr(F, FLAGS.att_activation))

    dim = self.pooling.output_size
    if FLAGS.use_type_emb:
      num_types = 2
      type_emb_dim = 10
      self.type_embedding = nn.Embedding(num_types, type_emb_dim)
      dim += type_emb_dim

    if FLAGS.use_answer_emb:
      self.context_dense = nn.Linear(self.pooling.output_size, FLAGS.emb_dim)
      self.answer_dense = nn.Linear(self.pooling.output_size, FLAGS.emb_dim)
      dim += 3

      self.logits = nn.Linear(dim, NUM_CLASSES)
      self.logits2 = nn.Linear(dim, NUM_CLASSES)


  def forward(self, inputs):
    x1 = inputs['passage']
    x2 = inputs['query']
    batch_size = x1.size(0)

    # TODO understand pytorch mask, not set to no pad
    x1_mask = get_mask(x1)
    x2_mask = get_mask(x2)

    #----- well here currently not reccurent dropout and not share dropout
    # Encode document with RNN
    input = {
              'content': inputs['passage'], 
              'char': inputs['passage_char'],
              'pos': inputs['passage_pos']
            }
    c = self.encode(input, x1_mask)
    
    input = {
              'content': inputs['query'], 
              'char': inputs['query_char'],
              'pos': inputs['query_pos']
            }
    # Encode question with RNN
    q = self.encode(input, x2_mask)

    # Align and aggregate
    c_check = c
    for i in range(FLAGS.hop):
        q_tilde = self.interactive_aligners[i].forward(c_check, q, x2_mask)
        c_bar = self.interactive_SFUs[i].forward(c_check, torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))
        c_tilde = self.self_aligners[i].forward(c_bar, x1_mask)
        c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
        c_check = self.aggregate_rnns[i].forward(c_hat, x1_mask)
    
    c = self.pooling(c_check, x1_mask)

    if FLAGS.use_type_emb:
      c = torch.cat([c, self.type_embedding(inputs['type'])], 1)

    if FLAGS.use_answer_emb:
      x1 = x = c

      neg = inputs['candidate_neg']
      pos = inputs['candidate_pos']
      na = inputs['candidate_na']

      neg_mask = get_mask(neg)
      pos_mask = get_mask(pos)
      na_mask = get_mask(na)

      input = {
              'content': inputs['candidate_neg'], 
              'char': inputs['candidate_neg_char'],
              'pos': inputs['candidate_neg_pos']
              }        
      neg = self.encode(input, neg_mask)
      input = {
              'content': inputs['candidate_pos'], 
              'char': inputs['candidate_pos_char'],
              'pos': inputs['candidate_pos_pos']
              }    
      pos = self.encode(input, pos_mask)
      input = {
              'content': inputs['candidate_na'], 
              'char': inputs['candidate_na_char'],
              'pos': inputs['candidate_na_pos']
              }    
      na = self.encode(input, na_mask)

      neg = self.pooling(neg, neg_mask)
      pos = self.pooling(pos, pos_mask)
      na = self.pooling(na, na_mask)

      answer = torch.stack([neg, pos, na], 1)

      # [batch_size, emb_dim]
      x = self.context_dense(x)
      # [batch_size, 3, emb_dim]
      answer = self.answer_dense(answer)
      x = answer.bmm(x.unsqueeze(1).transpose(2, 1))
      x = x.view(batch_size, NUM_CLASSES)

      x = torch.cat([x1, x], -1)
      c = x

    x = self.logits(c)

    if FLAGS.split_type:
      x2 = self.logits2(c)
      mask = torch.eq(inputs['type'], 0).float().unsqueeze(1)
      x = x * mask + x2 * (1 - mask)

    return x


