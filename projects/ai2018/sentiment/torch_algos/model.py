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
import gezi
import os

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
  def __init__(self, embedding=None):
    super(Model, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    Rnn = lele.layers.StackedBRNN if not FLAGS.torch_cudnn_rnn else lele.layers.CudnnRnn 
    
    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.dropout = nn.Dropout(p=(1 - FLAGS.keep_prob))

    emb_dim = FLAGS.emb_dim 

    self.embedding = wenzheng.pyt.get_embedding(vocab_size, 
                                                emb_dim, 
                                                embedding if embedding is not None else FLAGS.word_embedding_file, 
                                                FLAGS.finetune_word_embedding)

    if FLAGS.use_char:
      char_vocab_file = FLAGS.vocab.replace('vocab.txt', 'char_vocab.txt')
      if FLAGS.use_char_emb:
        assert os.path.exists(char_vocab_file) 
        char_vocab = gezi.Vocabulary(char_vocab_file, min_count=FLAGS.char_min_count)
        logging.info('using char vocab:', char_vocab_file, 'size:', char_vocab.size())
        self.char_embedding = wenzheng.pyt.get_embedding(char_vocab.size(), 
                                                        FLAGS.char_emb_dim, 
                                                        FLAGS.word_embedding_file.replace('emb.npy', 'char_emb.npy') if FLAGS.word_embedding_file else None,  
                                                        trainable=FLAGS.finetune_char_embedding)
      else:
        self.char_embedding = self.embedding

    if FLAGS.use_char:
      self.char_encode = lele.layers.StackedBRNN(
            input_size=FLAGS.char_emb_dim,
            hidden_size=self.num_units,
            num_layers=1,
            dropout_rate=1 - FLAGS.keep_prob,
            dropout_output=False,
            recurrent_dropout=FLAGS.recurrent_dropout,
            concat_layers=False,
            rnn_type=FLAGS.cell,
            padding=FLAGS.rnn_padding,
        )    

      self.char_pooling = lele.layers.Pooling(FLAGS.char_output_method, input_size= 2 * self.num_units)
      if FLAGS.char_combiner == 'sfu':
        self.char_fc = nn.Linear(2 * self.num_units, emb_dim)
        self.char_sfu_combine = lele.layers.SFUCombiner(emb_dim, 3 * emb_dim, dropout_rate=1 - FLAGS.keep_prob)
        encode_input_size = emb_dim
      else:
        # concat
        encode_input_size = emb_dim + 2 * self.num_units
    else:
      encode_input_size = emb_dim

    if FLAGS.use_pos:
      pos_vocab_file = FLAGS.vocab.replace('vocab.txt', 'pos_vocab.txt')
      assert os.path.exists(pos_vocab_file)
      pos_vocab = gezi.Vocabulary(pos_vocab_file, min_count=FLAGS.tag_min_count)
      logging.info('using pos vocab:', pos_vocab_file, 'size:', pos_vocab.size())
      self.pos_embedding = wenzheng.pyt.get_embedding(pos_vocab.size(), FLAGS.tag_emb_dim)
      encode_input_size += FLAGS.tag_emb_dim

    if FLAGS.use_ner:
      ner_vocab_file = FLAGS.vocab.replace('vocab.txt', 'ner_vocab.txt')
      assert os.path.exists(ner_vocab_file)
      ner_vocab = gezi.Vocabulary(ner_vocab_file, min_count=FLAGS.tag_min_count)
      logging.info('using ner vocab:', ner_vocab_file, 'size:', ner_vocab.size())
      self.ner_embedding = wenzheng.pyt.get_embedding(ner_vocab.size(), FLAGS.tag_emb_dim)
      encode_input_size += FLAGS.tag_emb_dim

    self.encode = Rnn(
            input_size=encode_input_size,
            hidden_size=self.num_units,
            num_layers=self.num_layers,
            dropout_rate=1 - FLAGS.keep_prob,
            dropout_output=False,
            recurrent_dropout=FLAGS.recurrent_dropout,
            concat_layers=FLAGS.concat_layers,
            rnn_type=FLAGS.cell,
            padding=FLAGS.rnn_padding,
        )    

    factor = self.num_layers if FLAGS.concat_layers else 1
    input_size = 2 * self.num_units * factor

    if FLAGS.use_label_att:
      self.label_emb_height = NUM_CLASSES if not FLAGS.label_emb_height else FLAGS.label_emb_height
      self.label_embedding = nn.Embedding(self.label_emb_height, FLAGS.emb_dim)

      self.att_dot_attentions = nn.ModuleList()
      self.att_encodes = nn.ModuleList()
      for i in range(FLAGS.label_hop):
        self.att_dot_attentions.append(lele.layers.DotAttention(input_size=input_size, 
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

        input_size = 2 * self.num_units

    if FLAGS.use_self_match:
      self.match_dot_attention = lele.layers.DotAttention(input_size=input_size, 
                                                          input_size2=input_size, 
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
                        input_size=input_size,
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
    max_c_len = x.size(1)

    if FLAGS.rnn_no_padding:
      x_mask = torch.zeros_like(x, dtype=torch.uint8)

    x = self.embedding(x)
  
      # TODO
    if FLAGS.use_char:
      cx = input['char']
      cx = cx.view(batch_size * max_c_len, FLAGS.char_limit)
      if FLAGS.char_padding:
        # HACK for pytorch rnn not allow all 0, TODO Too slow...
        cx = torch.cat([torch.ones([batch_size * max_c_len, 1], dtype=torch.int64).cuda(), cx], 1)
        cx_mask = cx.eq(0)
      else:
        cx_mask = torch.zeros_like(cx, dtype=torch.uint8)
      cx = self.char_embedding(cx)
      cx = self.char_encode(cx, cx_mask)
      cx = self.char_pooling(cx, cx_mask)
      cx = cx.view(batch_size, max_c_len, 2 * self.num_units)

      if FLAGS.char_combiner == 'concat':
        x = torch.cat([x, cx], 2)
      elif FLAGS.char_combiner == 'sfu':
        cx = self.char_fc(cx)
        x = self.char_sfu_combine(x, cx)

    if FLAGS.use_pos:
      px = input['pos']
      px = self.pos_embedding(px)
      x = torch.cat([x, px], 2)

    if FLAGS.use_ner:
      nx = input['ner']
      nx = self.ner_embedding(nx)
      x = torch.cat([x, nx], 2)

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
# NOTICE mainly use this one
# TODO word and char encoder can merge to one Encoder handle inputs with word and char
class MReader(nn.Module):
  def __init__(self, embedding=None):
    super(MReader, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    emb_dim = FLAGS.emb_dim 

    self.embedding = wenzheng.pyt.get_embedding(vocab_size, 
                                                emb_dim, 
                                                embedding if embedding is not None else FLAGS.word_embedding_file, 
                                                FLAGS.finetune_word_embedding)

    if FLAGS.use_char:
      char_vocab_file = FLAGS.vocab.replace('vocab.txt', 'char_vocab.txt')
      if FLAGS.use_char_emb:
        assert os.path.exists(char_vocab_file) 
        char_vocab = gezi.Vocabulary(char_vocab_file, min_count=FLAGS.char_min_count)
        logging.info('using char vocab:', char_vocab_file, 'size:', char_vocab.size())
        self.char_embedding = wenzheng.pyt.get_embedding(char_vocab.size(), 
                                                        FLAGS.char_emb_dim, 
                                                        FLAGS.word_embedding_file.replace('emb.npy', 'char_emb.npy') if FLAGS.word_embedding_file else None,  
                                                        trainable=FLAGS.finetune_char_embedding)
      else:
        self.char_embedding = self.embedding


    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.dropout = nn.Dropout(p=(1 - FLAGS.keep_prob))

    Rnn = lele.layers.StackedBRNN if not FLAGS.torch_cudnn_rnn else lele.layers.CudnnRnn

    if FLAGS.use_char:
      if FLAGS.char_encoder == 'rnn':
        self.char_encode = lele.layers.StackedBRNN(
              input_size=FLAGS.char_emb_dim,
              hidden_size=self.num_units,
              num_layers=1,
              dropout_rate=1 - FLAGS.keep_prob,
              dropout_output=False,
              recurrent_dropout=FLAGS.recurrent_dropout,
              concat_layers=False,
              rnn_type=FLAGS.cell,
              padding=FLAGS.rnn_padding,
          )    
        self.char_pooling = lele.layers.Pooling(FLAGS.char_output_method, input_size= 2 * self.num_units)
      elif FLAGS.char_encoder == 'conv':
        # seems slow and easy to overfit compare to rnn
        config = {
          "encoder": {
            "name": "elmo",
            "projection_dim": 2 * self.num_units, 
            "cell_clip": 3, 
            "proj_clip": 3,
            "dim": 4096,
            "n_layers": 2
            },
          "token_embedder": {
            "name": "cnn",
            "activation": "relu",
            #"filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]],
            #"filters": [[1, 32], [2, 32]],
            "filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512]],
            "n_highway": 2, 
            "char_dim": 300,
            "max_characters_per_token": 6
          },
        }
        self.char_encode = lele.layers.ConvTokenEmbedder(config, None, self.char_embedding, True)
      else:
        raise ValueError(FLAGS.char_encoder)
      
      if FLAGS.char_combiner == 'sfu':
        self.char_fc = nn.Linear(2 * self.num_units, emb_dim)
        self.char_sfu_combine = lele.layers.SFUCombiner(emb_dim, 3 * emb_dim, dropout_rate=1 - FLAGS.keep_prob)
        encode_input_size = emb_dim
      else:
        # concat
        encode_input_size = emb_dim + 2 * self.num_units
    else:
      encode_input_size = emb_dim

    if FLAGS.use_pos:
      pos_vocab_file = FLAGS.vocab.replace('vocab.txt', 'pos_vocab.txt')
      assert os.path.exists(pos_vocab_file)
      pos_vocab = gezi.Vocabulary(pos_vocab_file, min_count=FLAGS.tag_min_count)
      logging.info('using pos vocab:', pos_vocab_file, 'size:', pos_vocab.size())
      self.pos_embedding = wenzheng.pyt.get_embedding(pos_vocab.size(), FLAGS.tag_emb_dim)
      encode_input_size += FLAGS.tag_emb_dim

    if FLAGS.use_ner:
      ner_vocab_file = FLAGS.vocab.replace('vocab.txt', 'ner_vocab.txt')
      assert os.path.exists(ner_vocab_file)
      ner_vocab = gezi.Vocabulary(ner_vocab_file, min_count=FLAGS.tag_min_count)
      logging.info('using ner vocab:', ner_vocab_file, 'size:', ner_vocab.size())
      self.ner_embedding = wenzheng.pyt.get_embedding(ner_vocab.size(), FLAGS.tag_emb_dim)
      encode_input_size += FLAGS.tag_emb_dim

    if FLAGS.encoder_type == 'elmo':
      config = {
          "encoder": {
            "name": "elmo",
            "projection_dim": encode_input_size, 
            "cell_clip": 3, 
            "proj_clip": 3,
            "dim": self.num_units,
            "n_layers": 2
            },
          "dropout": 0.3
      }
      self.encode = lele.layers.ElmobiLm(config, True)
      doc_hidden_size = 2 * encode_input_size
      self.elmo_fc = nn.Linear(2 * encode_input_size, 2 * self.num_units)
    elif FLAGS.encoder_type == 'transformer':
      doc_hidden_size = 256
      self.transformer_fc = nn.Linear(encode_input_size, doc_hidden_size)
      self.encode = lele.layers.transformer.get_encoder(vocab_size, d_model=doc_hidden_size, d_ff=256, N=2, h=4)
    else:
      self.encode = Rnn(
              input_size=encode_input_size,
              hidden_size=self.num_units,
              num_layers=self.num_layers,
              dropout_rate=1 - FLAGS.keep_prob,
              dropout_output=False,
              recurrent_dropout=FLAGS.recurrent_dropout,
              concat_layers=FLAGS.concat_layers,
              rnn_type=FLAGS.cell,
              padding=FLAGS.rnn_padding,
          )    
      doc_hidden_size = 2 * self.num_units

    assert not FLAGS.concat_layers
    
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

    self.vocab_size = vocab_size 
    self.doc_hidden_size = doc_hidden_size

    # for lm and load simple...
    #if FLAGS.lm_model:
    self.hidden2tag = nn.Linear(self.num_units, self.vocab_size - 1)

  def forward(self, input, training=False):
    #print('------------', input['source'])
    x = input['content'] 
    #print(x.shape)
    x_mask = x.eq(0)
    batch_size = x.size(0)
    max_c_len = x.size(1)

    if FLAGS.rnn_no_padding:
      x_mask = torch.zeros_like(x, dtype=torch.uint8)

    x = self.embedding(x)

    if FLAGS.use_char:
      cx = input['char']
      if FLAGS.char_encoder == 'rnn':
        cx = cx.view(batch_size * max_c_len, FLAGS.char_limit)
        if FLAGS.char_padding:
          # HACK for pytorch rnn not allow all 0, TODO Too slow...
          cx = torch.cat([torch.ones([batch_size * max_c_len, 1], dtype=torch.int64).cuda(), cx], 1)
          cx_mask = cx.eq(0)
        else:
          cx_mask = torch.zeros_like(cx, dtype=torch.uint8)

        cx = self.char_embedding(cx)
        cx = self.char_encode(cx, cx_mask)
        cx = self.char_pooling(cx, cx_mask)
        cx = cx.view(batch_size, max_c_len, 2 * self.num_units)
      elif FLAGS.char_encoder == 'conv':
        cx = self.char_encode(None, cx, (batch_size, max_c_len)) 
      else:
        raise ValueError(FLAGS.char_encoder)

      if FLAGS.char_combiner == 'concat':
        x = torch.cat([x, cx], 2)
      elif FLAGS.char_combiner == 'sfu':
        cx = self.char_fc(cx)
        x = self.char_sfu_combine(x, cx)

    if FLAGS.use_pos:
      px = input['pos']
      px = self.pos_embedding(px)
      x = torch.cat([x, px], 2)

    if FLAGS.use_ner:
      nx = input['ner']
      nx = self.ner_embedding(nx)
      x = torch.cat([x, nx], 2)

    if FLAGS.encoder_type == 'elmo':
      # note HIT elmo mask different from HKUST mask.. need 1 -
      x = self.encode(x, 1 - x_mask)
      x = x[0]
      x = self.elmo_fc(x)
    elif FLAGS.encoder_type == 'transformer':
      x = self.transformer_fc(x)
      x = self.encode(x, 1 - x_mask)
      #print(x.shape)
    else:
      x = self.encode(x, x_mask)

    if FLAGS.lm_model:
      return x

    #print('x---------------', x.shape)

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
    
    #print(c_check.shape, q.shape, x2_mask.shape)
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

  
