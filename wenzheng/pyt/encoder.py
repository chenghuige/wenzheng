#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   encoder.py
#        \author   chenghuige  
#          \date   2018-09-29 07:39:50.763448
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import torch
from torch import nn

import gezi
logging = gezi.logging

import lele
import wenzheng

class TextEncoder(nn.Module):
  """
  Bidirectional Encoder 
  can be used for Language Model and also for text classification or others 
  input is batch of sentence ids [batch_size, num_steps]
  output is [batch_size, num_steps, 2 * hidden_dim]
  for text classification you can use pooling to get [batch_size, dim] as text resprestation
  for language model you can just add fc layer to convert 2 * hidden_dim to vocab_size -1 and calc cross entropy loss
  Notice you must outputs hidden_dim(forward) and hidden_dim(back_ward) concated at last dim as 2 * hidden dim, so MUST be bidirectional
  """
  def __init__(self,
               config, 
               embedding_weight=None,
               use_char=False,
               use_char_emb=True,
               use_pos=False,
               use_ner=False,
               lm_model=False,
              ):
    super(TextEncoder, self).__init__()

    Rnn = lele.layers.StackedBRNN

    word_config = config['word']
    vocab_file = word_config['vocab']
    vocab = gezi.Vocabulary(vocab_file)
    vocab_size = vocab.size()
    word_config['vocab_size'] = vocab_size
    num_layers = word_config['num_layers']
    hidden_size = word_config['hidden_size']
    emb_dim = word_config['emb_dim']
    word_embedding_file = word_config['embedding_file']
    finetune_word_embedding = word_config['trainable']
    embedding_weight = embedding_weight if embedding_weight is not None else word_embedding_file

    self.embedding = wenzheng.pyt.get_embedding(vocab_size, 
                                                emb_dim, 
                                                embedding_weight, 
                                                trainable=finetune_word_embedding)

    self.char_embedding = None
    if use_char:
      char_config = config['char']
      char_config['use_char_emb'] = use_char_emb
      if use_char_emb:
        char_vocab_file = vocab_file.replace('vocab.txt', 'char_vocab.txt')
        assert os.path.exists(char_vocab_file)
        char_config['vocab'] = char_vocab_file
        char_vocab = gezi.Vocabulary(char_vocab_file)
        char_vocab_size = char_vocab.size()
        char_config['char_vocab_size'] = char_vocab_size
        char_emb_dim = char_config['emb_dim']
        char_embedding_weight = word_embedding_file.replace('emb.npy', 'char_emb.npy') if word_embedding_file else None
        char_config['embedding_weight'] = char_embedding_weight
        finetune_char_embedding = char_config['trainable']
        self.char_embedding = wenzheng.pyt.get_embedding(char_vocab_size, 
                                                         char_emb_dim, 
                                                         char_embedding_weight,  
                                                         trainable=finetune_char_embedding)
      else:
        self.char_embedding = self.embedding

    dropout_rate = config['dropout_rate']
    recurrent_dropout = config['recurrent_dropout']
    cell = config['cell']
    rnn_padding = config['rnn_padding']
    rnn_no_padding = config['rnn_no_padding']
    concat_layers = config['concat_layers']

    if use_char:
      char_hidden_size = char_config['hidden_size']
      self.char_hidden_size = char_hidden_size
      char_output_method = char_config['output_method']
      char_combiner = char_config['combiner']
      self.char_combiner = char_combiner
      char_padding = char_config['padding']
      self.char_padding = char_padding
      self.char_limit = char_config['limit']
      self.char_encode = Rnn(
            input_size=char_emb_dim,
            hidden_size=char_hidden_size,
            num_layers=1,
            dropout_rate=1 - dropout_rate,
            dropout_output=False,
            recurrent_dropout=recurrent_dropout,
            concat_layers=False,
            rnn_type=cell,
            padding=rnn_padding,
        )    

      self.char_pooling = lele.layers.Pooling(char_output_method, input_size= 2 * char_hidden_size)
      if char_combiner == 'sfu':
        self.char_fc = nn.Linear(2 * char_hidden_size, emb_dim)
        self.char_sfu_combine = lele.layers.SFUCombiner(emb_dim, 3 * emb_dim, dropout_rate=dropout_rate)
        encode_input_size = emb_dim
      else:
        # concat
        encode_input_size = emb_dim + 2 * char_hidden_size
    else:
      encode_input_size = emb_dim

    self.pos_embedding = None
    if use_pos:
      pos_config = config['pos']
      tag_emb_dim = pos_config['emb_dim']
      pos_vocab_file = vocab_file.replace('vocab.txt', 'pos_vocab.txt')
      assert os.path.exists(pos_vocab_file)
      pos_config['vocab'] = pos_vocab_file
      pos_vocab = gezi.Vocabulary(pos_vocab_file)
      self.pos_embedding = wenzheng.pyt.get_embedding(pos_vocab.size(), tag_emb_dim)
      encode_input_size += tag_emb_dim

    self.ner_embedding = None
    if use_ner:
      ner_config = config['ner']
      tag_emb_dim = ner_config['emb_dim']
      ner_vocab_file = vocab_file.replace('vocab.txt', 'ner_vocab.txt')
      assert os.path.exists(ner_vocab_file)
      ner_config['vocab'] = ner_vocab_file
      ner_vocab = gezi.Vocabulary(ner_vocab_file)
      self.ner_embedding = wenzheng.pyt.get_embedding(ner_vocab.size(), tag_emb_dim)
      encode_input_size += tag_emb_dim

    hidden_size = word_config['hidden_size']
    num_layers = word_config['num_layers']
    self.encode = Rnn(
            input_size=encode_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            dropout_output=False,
            recurrent_dropout=recurrent_dropout,
            concat_layers=concat_layers,
            rnn_type=cell,
            padding=rnn_padding,
        )    

    factor = num_layers if concat_layers else 1
    output_size = 2 * hidden_size * factor

    self.vocab_size = vocab_size
    self.output_size = output_size
    self.rnn_no_padding = rnn_no_padding
    self.use_char = use_char 
    self.use_pos, self.use_ner = use_pos, use_ner

    if lm_model:
      # -1 for excluding padding  0
      self.hidden2tag = nn.Linear(hidden_size * factor, self.vocab_size - 1)
    
    self.lm_model = lm_model

    try:
      import yaml 
      logging.info('config\n', yaml.dump(config, default_flow_style=False))
    except Exception:
      logging.info('config', config)

  def get_mask(self, x):
    if self.rnn_no_padding:
      x_mask = torch.zeros_like(x, dtype=torch.uint8)   
    else:
      x_mask = x.eq(0)

    return x_mask

  # TODO training not needed, since pytorch has model.eval model.train here just compact for tensorflow
  def forward(self, input, mask=None, training=False):
    assert isinstance(input, dict)
    x = input['content'] 
    #print(x.shape)
    #print(input['source'])

    x_mask = mask if mask is not None else self.get_mask(x)
    batch_size = x.size(0)
    max_c_len = x.size(1)

    x = self.embedding(x)

    if self.use_char:
      cx = input['char']
      cx = cx.view(batch_size * max_c_len, self.char_limit)
      if self.char_padding:
        # HACK for pytorch rnn not allow all 0, TODO Too slow...
        cx = torch.cat([torch.ones([batch_size * max_c_len, 1], dtype=torch.int64).cuda(), cx], 1)
        cx_mask = cx.eq(0)
      else:
        cx_mask = torch.zeros_like(cx, dtype=torch.uint8)

      cx = self.char_embedding(cx)
      cx = self.char_encode(cx, cx_mask)
      cx = self.char_pooling(cx, cx_mask)
      cx = cx.view(batch_size, max_c_len, 2 * self.char_hidden_size)

      if self.char_combiner == 'concat':
        x = torch.cat([x, cx], 2)
      elif self.char_combiner == 'sfu':
        cx = self.char_fc(cx)
        x = self.char_sfu_combine(x, cx)
      else:
        raise ValueError(self.char_combiner)

    if self.use_pos:
      px = input['pos']
      px = self.pos_embedding(px)
      x = torch.cat([x, px], 2)

    if self.use_ner:
      nx = input['ner']
      nx = self.ner_embedding(nx)
      x = torch.cat([x, nx], 2)

    x = self.encode(x, x_mask)

    return x