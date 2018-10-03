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

import melt
logging = melt.logging

def get_mask(x):
    if FLAGS.rnn_no_padding:
        x_mask = torch.zeros_like(x, dtype=torch.uint8)
    else:
        x_mask = x.eq(0)
    return x_mask

#v3 change to use lele.layers.CudnnRnn and recurrent + share dropout
class MnemonicReaderV3(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
    def __init__(self, args=None):
        super(MnemonicReaderV3, self).__init__()
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
        self.dropout_rate = 1 - args.keep_prob

        self.num_layers = 1

        # Encoder
        self.encoding_rnn = layers.CudnnRnn(
            input_size=doc_input_size,
            hidden_size=args.rnn_hidden_size,
            num_layers=1,
            dropout_rate=1 - args.keep_prob,
            dropout_output=False,
            concat_layers=False,
            rnn_type=self.RNN_TYPES['gru'],
            padding=args.rnn_padding,
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
        
        self.pooling = lele.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        input_size= 2 * args.rnn_hidden_size,
                        top_k=FLAGS.top_k, 
                        att_activation=getattr(F, FLAGS.att_activation))

        pre_logits_dim = self.pooling.output_size
        if FLAGS.use_type_emb:
            num_types = 2
            type_emb_dim = 10
            self.type_embedding = nn.Embedding(num_types, type_emb_dim)
            pre_logits_dim += type_emb_dim
            
        self.logits = nn.Linear(pre_logits_dim, NUM_CLASSES)
        self.logits2 = nn.Linear(pre_logits_dim, NUM_CLASSES)


    def forward(self, inputs):
        x1 = inputs['passage']
        x2 = inputs['query']

        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # TODO understand pytorch mask, not set to no pad
        x1_mask = get_mask(x1)
        x2_mask = get_mask(x2)

        num_units = self.encoding_rnn.num_units
        batch_size = x1.size(0)
        mask_fws = [F.dropout(torch.ones(1, batch_size, num_units[layer]).cuda(),p=self.dropout_rate, training=self.training) for layer in range(self.num_layers)]
        mask_bws = [F.dropout(torch.ones(1, batch_size, num_units[layer]).cuda(),p=self.dropout_rate, training=self.training) for layer in range(self.num_layers)]

        # Encode document with RNN
        c = self.encoding_rnn(x1_emb, x1_mask, mask_fws, mask_bws)
        
        # Encode question with RNN
        q = self.encoding_rnn(x2_emb, x2_mask, mask_fws, mask_bws)

        # Align and aggregate
        c_check = c
        for i in range(self.args.hop):
            q_tilde = self.interactive_aligners[i].forward(c_check, q, x2_mask)
            c_bar = self.interactive_SFUs[i].forward(c_check, torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))
            c_tilde = self.self_aligners[i].forward(c_bar, x1_mask)
            c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
            c_check = self.aggregate_rnns[i].forward(c_hat, x1_mask)
        
        if self.args.pooling_no_padding:
            #logging.info('----------pooling no padding!')
            x1_mask = torch.zeros_like(x1, dtype=torch.uint8)

        c = self.pooling(c_check, x1_mask)

        if FLAGS.use_type_emb:
            c = torch.cat([c, self.type_embedding(inputs['type'])], 1)
    
        x = self.logits(c)

        if self.args.split_type:
            x2 = self.logits2(c)
            mask = torch.eq(inputs['type'], 0).float().unsqueeze(1)
            x = x * mask + x2 * (1 - mask)

        return x

# MnemonicReader is actually V2, v2 support more pooling, v2 with split got 7368
# TODO err why wrong.. slow at beggining FIXME can just fall back to V1
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
            padding=args.rnn_padding,
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
        
        self.pooling = lele.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        input_size= 2 * args.rnn_hidden_size,
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

        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # TODO understand pytorch mask, not set to no pad
        x1_mask = get_mask(x1)
        x2_mask = get_mask(x2)

        #----- well here currently not reccurent dropout and not share dropout
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
        
        if self.args.pooling_no_padding:
            #logging.info('----------pooling no padding!')
            x1_mask = torch.zeros_like(x1, dtype=torch.uint8)

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

            neg_emb = self.embedding(neg)
            pos_emb = self.embedding(pos)
            na_emb = self.embedding(na)

            neg = self.encoding_rnn(neg_emb, neg_mask)
            pos = self.encoding_rnn(pos_emb, pos_mask)
            na = self.encoding_rnn(na_emb, na_mask)

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

        if self.args.split_type:
            x2 = self.logits2(c)
            mask = torch.eq(inputs['type'], 0).float().unsqueeze(1)
            x = x * mask + x2 * (1 - mask)

        return x


# this is V1 which already got good results 736, V2 will have more pooling method with mask support, also add support for recurrent dropout..
class MnemonicReaderV1(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
    def __init__(self, args=None):
        super(MnemonicReaderV1, self).__init__()
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
