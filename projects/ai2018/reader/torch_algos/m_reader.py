#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   m_reader.py
#        \author   chenghuige  
#          \date   2018-09-28 11:15:19.088911
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


"""Implementation of the Mnemonic Reader."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lele
layers = lele.layers
from torch.autograd import Variable


import wenzheng
from wenzheng.utils import vocabulary

from algos.config import NUM_CLASSES

import numpy as np


# this is V3, V4 will have more pooling method with mask support, also add support for recurrent dropout..
class MnemonicReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
    def __init__(self, args=None):
        super(MnemonicReader, self).__init__()
        if args is None:
          args = FLAGS
        # Store config
        self.args = args
    
        vocabulary.init()
        vocab_size = vocabulary.get_vocab_size() 
        
        self.embedding = wenzheng.pyt.get_embedding(vocab_size, 
                                                    args.emb_dim, 
                                                    args.word_embedding_file, 
                                                    args.finetune_word_embedding)

        doc_input_size = args.emb_dim 

        # Encoder
        self.encoding_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.rnn_hidden_size,
            num_layers=1,
            dropout_rate=1 - args.keep_prob,
            dropout_output=False,
            concat_layers=False,
            rnn_type=self.RNN_TYPES['gru'],
            padding=False,
        )

        doc_hidden_size = 2 * args.rnn_hidden_size
        
        # Interactive aligning, self aligning and aggregating
        self.interactive_aligners = nn.ModuleList()
        self.interactive_SFUs = nn.ModuleList()
        self.self_aligners = nn.ModuleList()
        self.self_SFUs = nn.ModuleList()
        self.aggregate_rnns = nn.ModuleList()
        
        for i in range(args.hop):
            # interactive aligner
            self.interactive_aligners.append(layers.SeqAttnMatch(doc_hidden_size, identity=True))
            self.interactive_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
            # self aligner
            self.self_aligners.append(layers.SelfAttnMatch(doc_hidden_size, identity=True, diag=False))
            self.self_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
            # aggregating
            self.aggregate_rnns.append(
                layers.StackedBRNN(
                    input_size=doc_hidden_size,
                    hidden_size=args.rnn_hidden_size,
                    num_layers=1,
                    dropout_rate=1 - args.keep_prob,
                    dropout_output=False,
                    concat_layers=False,
                    rnn_type=self.RNN_TYPES['gru'],
                    padding=False,
                )
            )

        self.logits = nn.Linear(2 * args.rnn_hidden_size, NUM_CLASSES)
        self.logits2 = nn.Linear(2 * args.rnn_hidden_size, NUM_CLASSES)


    def forward(self, inputs):
        x1 = inputs['passage']
        x2 = inputs['query']

        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # x1_mask = x1 == 0
        # x2_mask = x2 == 0
        
        # TODO understand pytorch mask, not set to no pad
        x1_mask = torch.zeros_like(x1, dtype=torch.uint8)
        x2_mask = torch.zeros_like(x2, dtype=torch.uint8)

        # Encode document with RNN
        c = self.encoding_rnn(x1_emb, x1_mask)
        
        # Encode question with RNN
        q = self.encoding_rnn(x2_emb, x2_mask)

        # Align and aggregate
        c_check = c
        for i in range(self.args.hop):
            q_tilde = self.interactive_aligners[i].forward(c_check, q, x2_mask)
            c_bar = self.interactive_SFUs[i].forward(c_check, torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))
            c_tilde = self.self_aligners[i].forward(c_bar, x1_mask)
            c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
            c_check = self.aggregate_rnns[i].forward(c_hat, x1_mask)

        # TODO support other pooling layer like attention topk in torch or without mask ?
        c = torch.max(c_check, 1)[0]
        
        x = self.logits(c)

        if self.args.split_type:
            x2 = self.logits2(c)
            mask = torch.eq(inputs['type'], 0).float().unsqueeze(1)
            x = x * mask + x2 * (1 - mask)

        return x
