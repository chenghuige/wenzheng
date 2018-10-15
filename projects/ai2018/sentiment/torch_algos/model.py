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
layers = lele.layers

#ModelV2 using cudnnrnn, not to use
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
            recurrent_dropout=FLAGS.recurrent_dropout,
            bw_dropout=FLAGS.bw_dropout,
            dropout_output=False,
            concat_layers=FLAGS.concat_layers,
            rnn_type=FLAGS.cell,
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

# Model is like RNet! if you use label att and self match
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

    factor = self.num_layers if FLAGS.concat_layers else 1

    Rnn = lele.layers.StackedBRNN if not FLAGS.torch_cudnn_rnn else lele.layers.CudnnRnn
    self.encode = Rnn(
            input_size=emb_dim,
            hidden_size=self.num_units,
            num_layers=self.num_layers,
            dropout_rate=1 - FLAGS.keep_prob,
            dropout_output=False,
            recurrent_dropout=FLAGS.recurrent_dropout,
            concat_layers=FLAGS.concat_layers,
            rnn_type=FLAGS.cell,
            padding=FLAGS.rnn_padding,
        )    

    if FLAGS.use_label_att:
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      self.label_embedding = nn.Embedding(self.label_emb_height, FLAGS.emb_dim)

      self.att_dot_attentions = nn.ModuleList()
      self.att_encodes = nn.ModuleList()
      for i in range(FLAGS.label_hop):
        self.att_dot_attentions.append(lele.layers.DotAttention(input_size=2 * self.num_units * factor, 
                                                                input_size2=FLAGS.emb_dim,
                                                                hidden=self.num_units, 
                                                                dropout_rate=1 - FLAGS.keep_prob, 
                                                                combiner=FLAGS.att_combiner))
        self.att_encodes.append(Rnn(
              input_size=self.att_dot_attentions[-1].output_size,
              hidden_size=self.num_units,
              num_layers=1,
              dropout_rate=1 - FLAGS.keep_prob,
              dropout_output=False,
              concat_layers=False,
              rnn_type=FLAGS.cell,
              padding=FLAGS.rnn_padding,
          ))

    if FLAGS.use_self_match:
      self.match_dot_attention = lele.layers.DotAttention(input_size=2 * self.num_units, 
                                                          input_size2=2 * self.num_units, 
                                                          hidden=self.num_units, 
                                                          dropout_rate=1 - FLAGS.keep_prob, 
                                                          combiner=FLAGS.att_combiner)
      self.match_encode = Rnn(
            input_size=self.match_dot_attention.output_size,
            hidden_size=self.num_units,
            num_layers=1,
            dropout_rate=1 - FLAGS.keep_prob,
            dropout_output=False,
            concat_layers=False,
            rnn_type=FLAGS.cell,
            padding=FLAGS.rnn_padding,
        )    

    self.pooling = lele.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        input_size= 2 * self.num_units * factor,
                        top_k=FLAGS.top_k, 
                        att_activation=getattr(F, FLAGS.att_activation))

    # input dim not as convinient as tf..
    pre_logits_dim = self.pooling.output_size
    
    self.num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
    self.logits = nn.Linear(pre_logits_dim, NUM_ATTRIBUTES * self.num_classes)

  def forward(self, input, training=False):
    x = input['content'] 
    #print(x.shape)
    x_mask = x.eq(0)
    batch_size = x.size(0)

    if FLAGS.rnn_no_padding:
      x_mask = torch.zeros_like(x, dtype=torch.uint8)

    x = self.embedding(x)

    x = self.encode(x, x_mask)

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

    x = x.view([-1, NUM_ATTRIBUTES, self.num_classes])

    return x
  
# MReader with hop==1 can be viewed as rnet also
class MReader(nn.Module):
  def __init__(self):
    super(MReader, self).__init__()
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

    Rnn = lele.layers.StackedBRNN if not FLAGS.torch_cudnn_rnn else lele.layers.CudnnRnn
    self.encode = Rnn(
            input_size=emb_dim,
            hidden_size=self.num_units,
            num_layers=self.num_layers,
            dropout_rate=1 - FLAGS.keep_prob,
            dropout_output=False,
            recurrent_dropout=FLAGS.recurrent_dropout,
            concat_layers=FLAGS.concat_layers,
            rnn_type=FLAGS.cell,
            padding=FLAGS.rnn_padding,
        )    

    self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
    self.label_embedding = nn.Embedding(self.label_emb_height, FLAGS.emb_dim)
  
    assert not FLAGS.concat_layers
    doc_hidden_size = 2 * self.num_units

    # here linear better or another rnn is better ?
    if not FLAGS.use_label_rnn:
      self.label_forward = nn.Linear(FLAGS.emb_dim, doc_hidden_size)
    else:
      self.label_forward = Rnn(
            input_size=emb_dim,
            hidden_size=self.num_units,
            num_layers=1,
            dropout_rate=1 - FLAGS.keep_prob,
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
              dropout_rate=1 - FLAGS.keep_prob,
              dropout_output=False,
              concat_layers=False,
              rnn_type=FLAGS.cell,
              padding=FLAGS.rnn_padding,
          )
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
    #print(x.shape)
    x_mask = x.eq(0)
    batch_size = x.size(0)

    if FLAGS.rnn_no_padding:
      x_mask = torch.zeros_like(x, dtype=torch.uint8)

    x = self.embedding(x)

    x = self.encode(x, x_mask)

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
    
    for i in range(FLAGS.hop):
      q_tilde = self.interactive_aligners[i].forward(c_check, q, x2_mask)
      c_bar = self.interactive_SFUs[i].forward(c_check, torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))
      c_tilde = self.self_aligners[i].forward(c_bar, x_mask)
      c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
      c_check = self.aggregate_rnns[i].forward(c_hat, x_mask)
    
    x = c_check

    x = self.pooling(x, x_mask)
    
    x = self.logits(x)  

    x = x.view([-1, NUM_ATTRIBUTES, self.num_classes])

    return x

class Fastai(nn.Module):
  def __init__(self):
    super(Fastai, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    emb_dim = FLAGS.emb_dim 
    self.num_classes = NUM_CLASSES
    self.model = lele.fastai.text.classifier(vocab_size, NUM_ATTRIBUTES * self.num_classes, emb_sz=emb_dim,
                                             nl=FLAGS.num_layers,
                                             embedding_weight=FLAGS.word_embedding_file)

  def forward(self, input, training=False):
    x = input['content']
    # TODO ..
    #x = x.permute(1, 0)
    x = x.transpose(0, 1)
    x = self.model(x)
    
    x = x[0]
    x = x.view([-1, NUM_ATTRIBUTES, self.num_classes])

    return x

  
