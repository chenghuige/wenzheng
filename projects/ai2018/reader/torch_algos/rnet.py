#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnet.py
#        \author   chenghuige  
#          \date   2018-09-28 10:24:33.014425
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import lele
layers = lele.layers

import wenzheng
from wenzheng.utils import vocabulary

from algos.config import NUM_CLASSES

import numpy as np

class Rnet(nn.Module):
  RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
  CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
  def __init__(self, args=None):
    super(Rnet, self).__init__()
    # Store config
    if args is None:
      args = FLAGS
    self.args = args

    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    # Word embeddings (+1 for padding)
    self.embedding = nn.Embedding(vocab_size,
                                  args.emb_dim,
                                  padding_idx=0)

    if FLAGS.word_embedding_file:
        self.embedding.weight.data.copy_(torch.from_numpy(np.load(FLAGS.word_embedding_file)))
        if not FLAGS.finetune_word_embedding:
            self.embedding.weight.requires_grad = False

    doc_input_size = args.emb_dim 

    # Encoder
    self.encode_rnn = layers.StackedBRNN(
        input_size=doc_input_size,
        hidden_size=args.rnn_hidden_size,
        num_layers=args.num_layers,
        dropout_rate=1 - args.keep_prob,
        dropout_output=False,
        concat_layers=True,
        rnn_type=self.RNN_TYPES['gru'],
        padding=False,
    )

    # Output sizes of rnn encoder
    doc_hidden_size = 2 * args.rnn_hidden_size
    question_hidden_size = 2 * args.rnn_hidden_size
    
    #if args.concat_rnn_layers:
    doc_hidden_size *= args.num_layers
    question_hidden_size *= args.num_layers

    # Gated-attention-based RNN of the whole question
    self.question_attn = layers.SeqAttnMatch(question_hidden_size, identity=False)
    self.question_attn_gate = layers.Gate(doc_hidden_size + question_hidden_size)
    self.question_attn_rnn = layers.StackedBRNN(
        input_size=doc_hidden_size + question_hidden_size,
        hidden_size=args.rnn_hidden_size,
        num_layers=1,
        dropout_rate=1 - args.keep_prob,
        dropout_output=False,
        concat_layers=False,
        rnn_type=self.RNN_TYPES['gru'],
        padding=False,
    )

    question_attn_hidden_size = 2 * args.rnn_hidden_size

    # Self-matching-attention-baed RNN of the whole doc
    self.doc_self_attn = layers.SelfAttnMatch(question_attn_hidden_size, identity=False)
    self.doc_self_attn_gate = layers.Gate(question_attn_hidden_size + question_attn_hidden_size)
    self.doc_self_attn_rnn = layers.StackedBRNN(
        input_size=question_attn_hidden_size + question_attn_hidden_size,
        hidden_size=args.rnn_hidden_size,
        num_layers=1,
        dropout_rate=1 - args.keep_prob,
        dropout_output=False,
        concat_layers=False,
        rnn_type=self.RNN_TYPES['gru'],
        padding=False,
    )

    doc_self_attn_hidden_size = 2 * args.rnn_hidden_size

    self.doc_self_attn_rnn2 = layers.StackedBRNN(
        input_size=doc_self_attn_hidden_size,
        hidden_size=args.rnn_hidden_size,
        num_layers=1,
        dropout_rate=1 - args.keep_prob,
        dropout_output=False,
        concat_layers=False,
        rnn_type=self.RNN_TYPES['gru'],
        padding=False,
    )

    self.logits = nn.Linear(2 * args.rnn_hidden_size, NUM_CLASSES, bias=True)


  def forward(self, inputs):
    """Inputs:
    x1 = document word indices             [batch * len_d]
    x1_c = document char indices           [batch * len_d]
    x1_f = document word features indices  [batch * len_d * nfeat]
    x1_mask = document padding mask        [batch * len_d]
    x2 = question word indices             [batch * len_q]
    x2_c = document char indices           [batch * len_d]
    x1_f = document word features indices  [batch * len_d * nfeat]
    x2_mask = question padding mask        [batch * len_q]
    """
    x1 = inputs['passage']
    x2 = inputs['query']

    # x1_mask = x1 == 0
    # x2_mask = x2 == 0

    # TODO understand pytorch mask, not set to no pad
    x1_mask = torch.zeros_like(x1, dtype=torch.uint8)
    x2_mask = torch.zeros_like(x2, dtype=torch.uint8)

    # Embed both document and question
    x1_emb = self.embedding(x1)
    x2_emb = self.embedding(x2)

    # # Dropout on embeddings
    # if self.args.dropout_emb > 0:
    #   x1_emb = F.dropout(x1_emb, p=self.args.dropout_emb, training=self.training)
    #   x2_emb = F.dropout(x2_emb, p=self.args.dropout_emb, training=self.training)

    # Encode document with RNN
    c = self.encode_rnn(x1_emb, x1_mask)

    # Encode question with RNN
    q = self.encode_rnn(x2_emb, x2_mask)

    # Match questions to docs
    question_attn_hiddens = self.question_attn(c, q, x2_mask)
    rnn_input = self.question_attn_gate(torch.cat([c, question_attn_hiddens], 2))
    c = self.question_attn_rnn(rnn_input, x1_mask)

    # Match documents to themselves
    doc_self_attn_hiddens = self.doc_self_attn(c, x1_mask)
    rnn_input = self.doc_self_attn_gate(torch.cat([c, doc_self_attn_hiddens], 2))
    c = self.doc_self_attn_rnn(rnn_input, x1_mask)
    c = self.doc_self_attn_rnn2(c, x1_mask)

    c = torch.max(c, 1)[0]
    x = self.logits(c)

    return x 

