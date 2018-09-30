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

from algos.config import NUM_CLASSES, NUM_ATTRIBUTES

import wenzheng
from wenzheng.utils import vocabulary

import melt
logging = melt.logging

import numpy as np
import lele

class ModelV2(nn.Module):
  def __init__(self):
    super(ModelV2, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    emb_dim = FLAGS.emb_dim 

    self.embedding = wenzheng.pyt.get_embedding(vocab_size, 
                                                emb_dim, 
                                                FLAGS.word_embedding_file, 
                                                FLAGS.finetune_word_embedding)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.dropout = nn.Dropout(p=(1 - FLAGS.keep_prob))
    
    #self.encode = nn.GRU(input_size=emb_dim, hidden_size=self.num_units, batch_first=True, bidirectional=True)
    self.encode = lele.layers.CudnnRnn(
            input_size=emb_dim,
            hidden_size=self.num_units,
            num_layers=self.num_layers,
            dropout_rate=1 - FLAGS.keep_prob,
            dropout_output=False,
            concat_layers=False,
            rnn_type=nn.GRU,
            padding=FLAGS.rnn_padding,
        )    

    self.pooling = lele.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        input_size= 2 * self.num_units,
                        top_k=FLAGS.top_k, 
                        att_activation=getattr(F, FLAGS.att_activation))

    # input dim not as convinient as tf..
    pre_logits_dim = self.pooling.output_size
    
    self.num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
    self.logits = nn.Linear(pre_logits_dim, NUM_ATTRIBUTES * self.num_classes)

  def forward(self, input, training=False):
    x = input['content'] 
    x_mask = x.eq(0)

    if FLAGS.rnn_no_padding:
      x_mask = torch.zeros_like(x, dtype=torch.uint8)

    x = self.embedding(x)

    x = self.encode(x, x_mask)

    x = self.pooling(x, x_mask)
    
    x = self.logits(x)  

    x = x.view([-1, NUM_ATTRIBUTES, self.num_classes])

    return x

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    emb_dim = FLAGS.emb_dim 

    self.embedding = wenzheng.pyt.get_embedding(vocab_size, 
                                                emb_dim, 
                                                FLAGS.word_embedding_file, 
                                                FLAGS.finetune_word_embedding)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.dropout = nn.Dropout(p=(1 - FLAGS.keep_prob))
    
    #self.encode = nn.GRU(input_size=emb_dim, hidden_size=self.num_units, batch_first=True, bidirectional=True)
    self.encode = lele.layers.StackedBRNN(
            input_size=emb_dim,
            hidden_size=self.num_units,
            num_layers=self.num_layers,
            dropout_rate=1 - FLAGS.keep_prob,
            dropout_output=False,
            concat_layers=False,
            rnn_type=nn.GRU,
            padding=FLAGS.rnn_padding,
        )    

    self.pooling = lele.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        input_size= 2 * self.num_units,
                        top_k=FLAGS.top_k, 
                        att_activation=getattr(F, FLAGS.att_activation))

    # input dim not as convinient as tf..
    pre_logits_dim = self.pooling.output_size
    
    self.num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
    self.logits = nn.Linear(pre_logits_dim, NUM_ATTRIBUTES * self.num_classes)

  def forward(self, input, training=False):
    x = input['content'] 
    x_mask = x.eq(0)

    if FLAGS.rnn_no_padding:
      x_mask = torch.zeros_like(x, dtype=torch.uint8)

    x = self.embedding(x)

    x = self.encode(x, x_mask)

    x = self.pooling(x, x_mask)
    
    x = self.logits(x)  

    x = x.view([-1, NUM_ATTRIBUTES, self.num_classes])

    return x
  
