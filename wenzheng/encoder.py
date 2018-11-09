#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   encoder.pyt
#        \author   chenghuige  
#          \date   2018-09-20 07:04:55.883159
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
flags.DEFINE_string('bert_dir', None, '')
flags.DEFINE_string('bert_config_file', None, '')


import sys 
import os

from tensorflow import keras

import gezi
import melt
logging = melt.logging
import numpy as np

import wenzheng.utils.input_flags
import wenzheng.utils.rnn_flags

class Encoder(melt.Model):
  def __init__(self, type='gru', keep_prob=None):
    super(Encoder, self).__init__()
    
    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.keep_prob = keep_prob or FLAGS.keep_prob
    self.recurrent_dropout = FLAGS.recurrent_dropout
    self.bw_dropout = FLAGS.bw_dropout
    self.concat_layers = FLAGS.concat_layers
    self.residual_connect = FLAGS.encoder_residual_connect

    logging.info(f'encoder:{type}')
    logging.info('encoder recurrent dropout:{}'.format(self.recurrent_dropout))
    logging.info('encoder concat layers:{}'.format(self.concat_layers))
    logging.info('encoder residual connect:{}'.format(self.residual_connect))
    logging.info('encoder bw dropout:{}'.format(self.bw_dropout))
    logging.info('encoder num_layers:{}'.format(self.num_layers))
    logging.info('encoder num_units:{}'.format(self.num_units))
    logging.info('encoder keep_prob:{}'.format(self.keep_prob))

    def get_encode(type):
      if type == 'bow' or type == 'none':
        encode = None
      elif type == 'gru' or type == 'rnn' or type == 'lstm':
        if type == 'rnn':
          type = FLAGS.cell or 'gru'
        encode = melt.layers.CudnnRnn(num_layers=self.num_layers, 
                                      num_units=self.num_units, 
                                      keep_prob=self.keep_prob,
                                      share_dropout=False,
                                      recurrent_dropout=self.recurrent_dropout,
                                      concat_layers=self.concat_layers,
                                      bw_dropout=self.bw_dropout,
                                      residual_connect=self.residual_connect,
                                      train_init_state=FLAGS.rnn_train_init_state,
                                      cell=type)
      elif type == 'cnn' or type == 'convnet':
        logging.info('encoder num_filters:{}'.format(FLAGS.num_filters))
        num_layers = FLAGS.num_layers if not ',' in type else 4
        encode = melt.layers.ConvNet(num_layers=num_layers,
                                     num_filters=FLAGS.num_filters,
                                     keep_prob=FLAGS.keep_prob,
                                     use_position_encoding=FLAGS.use_position_encoding)
      elif type == 'qanet':
        num_layers = FLAGS.num_layers if not ',' in type else 4
        encode = melt.layers.QANet(num_layers=num_layers,
                                   num_filters=FLAGS.num_filters,
                                   keep_prob=0.5,
                                   kernel_size=3,
                                   num_heads=4)
      else:
        raise ValueError('not support {} now'.format(type))
      if encode is not None:
        encode.bidirectional = False
      if type == 'gru' or type == 'rnn' or type == 'lstm':
        encode.bidirectional = True
      return encode

    self.encodes = []
    for type in type.split(','):
      #print(type, get_encode(type))
      # TODO FIMXE tensorflow 1.1 fail layer_utils.py weights += layer.trainable_weights 'property' object is not iterable
      self.encodes.append(get_encode(type))

    logging.info(self.encodes)


  def call(self, seq, seq_len, mask_fws=None, mask_bws=None, training=False):
    for encode in self.encodes:
      if encode is None:
        if mask_fws is None:
          pass
        else:
          seq = seq * mask_fws[0]
        continue
      if encode.bidirectional:
        seq = encode(seq, seq_len, mask_fws, mask_fws, training=training)
      elif mask_fws is not None:
        seq = encode(seq, seq_len, mask_fws, training=training)
      else:
        seq = encode(seq, seq_len, training=training)
    return seq

#-----------------Both TexEncoder and BertEncoder are for pretraining then loading for finetune
class TextEncoder(melt.Model):
  """
  Bidirectional Encoder 
  can be used for Language Model and also for text classification or others 
  input is batch of sentence ids [batch_size, num_steps]
  output is [batch_size, num_steps, 2 * hidden_dim]
  for text classification you can use pooling to get [batch_size, dim] as text resprestation
  for language model you can just add fc layer to convert 2 * hidden_dim to vocab_size -1 and calc cross entropy loss
  Notice you must oututs hidden_dim(forward) and hidden_dim(back_ward) concated at last dim as 2 * hidden dim, so MUST be bidirectional
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

    Rnn = melt.layers.CudnnRnn

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
    num_finetune_words = word_config['num_finetune']
    embedding_weight = embedding_weight if embedding_weight is not None else word_embedding_file

    # vocab2_size = 0
    # if num_finetune_words:
    #   vocab2_size = vocab_size - num_finetune_words
    #   vocab_size = num_finetune_words
    
    # # TODO FIXME TypeError: Eager execution of tf.constant with unsupported shape (value has 2039400 elements, shape is (144299, 300) with 43289700 elements).
    # # For large vocab not Eager mode not ok..
    # self.embedding = wenzheng.Embedding(vocab_size, 
    #                                     emb_dim, 
    #                                     embedding_weight, 
    #                                     trainable=finetune_word_embedding,
    #                                     vocab2_size=vocab2_size)
    num_freeze_words = 0
    if num_finetune_words:
      num_freeze_words = vocab_size - num_finetune_words
    self.embedding = wenzheng.Embedding(vocab_size, 
                                        emb_dim, 
                                        embedding_weight, 
                                        trainable=finetune_word_embedding,
                                        freeze_size=num_freeze_words)

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
        num_finetune_chars = char_config['num_finetune']
        num_freeze_chars = 0
        if num_finetune_chars:
          num_freeze_chars = char_vocab_size - num_finetune_chars
        self.char_embedding = wenzheng.Embedding(char_vocab_size, 
                                                 char_emb_dim, 
                                                 char_embedding_weight, 
                                                 trainable=finetune_char_embedding,
                                                 freeze_size=num_freeze_chars)
      else:
        self.char_embedding = self.embedding

    dropout_rate = config['dropout_rate']
    self.keep_prob = 1 - dropout_rate
    recurrent_dropout = config['recurrent_dropout']
    cell = config['cell']
    rnn_padding = config['rnn_padding']
    rnn_no_padding = config['rnn_no_padding']
    concat_layers = config['concat_layers']
    train_init_state = config['rnn_train_init_state']

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
            num_layers=1,
            num_units=char_hidden_size,
            keep_prob=1 - dropout_rate,
            recurrent_dropout=recurrent_dropout,
            concat_layers=False,
            train_init_state=False if lm_model else train_init_state,
            cell=cell,
        )    

      self.char_pooling = melt.layers.Pooling(char_output_method)
      if char_combiner == 'sfu':
        self.char_sfu_combine = melt.layers.SemanticFusionCombine(keep_prob=self.keep_prob)

    self.pos_embedding = None
    if use_pos:
      pos_config = config['pos']
      tag_emb_dim = pos_config['emb_dim']
      pos_vocab_file = vocab_file.replace('vocab.txt', 'pos_vocab.txt')
      assert os.path.exists(pos_vocab_file)
      pos_config['vocab'] = pos_vocab_file
      pos_vocab = gezi.Vocabulary(pos_vocab_file)
      self.pos_embedding = wenzheng.Embedding(pos_vocab.size(), tag_emb_dim)

    self.ner_embedding = None
    if use_ner:
      ner_config = config['ner']
      tag_emb_dim = ner_config['emb_dim']
      ner_vocab_file = vocab_file.replace('vocab.txt', 'ner_vocab.txt')
      assert os.path.exists(ner_vocab_file)
      ner_config['vocab'] = ner_vocab_file
      ner_vocab = gezi.Vocabulary(ner_vocab_file)
      self.ner_embedding = wenzheng.Embedding(ner_vocab.size(), tag_emb_dim)

    hidden_size = word_config['hidden_size']
    num_layers = word_config['num_layers']

    if lm_model:
      assert config['encoder'] == 'rnn'

    if not lm_model and config['encoder'] != 'rnn':
      self.encode = Encoder(config['encoder'])   
    else:
      self.encode = Rnn(
            num_layers=num_layers,
            num_units=hidden_size,
            keep_prob=1 - dropout_rate,
            recurrent_dropout=recurrent_dropout,
            concat_layers=concat_layers,
            # just for simple finetune... since now init state var scope has some problem
            train_init_state=False,
            cell=cell,
      )

    factor = num_layers if concat_layers else 1
    output_size = 2 * hidden_size * factor

    self.vocab_size = vocab_size
    self.output_size = output_size
    self.rnn_no_padding = rnn_no_padding
    self.use_char = use_char 
    self.use_pos, self.use_ner = use_pos, use_ner
    self.num_units = hidden_size

    if lm_model:
      # -1 for excluding padding  0
      self.hidden2tag = keras.layers.Dense(self.vocab_size - 1)
    
    self.lm_model = lm_model

    try:
      import yaml 
      logging.info('config\n', yaml.dump(config, default_flow_style=False))
    except Exception:
      logging.info('config', config)

  # TODO training not needed, since pytorch has model.eval model.train here just compact for tensorflow
  def call(self, input, c_len=None, max_c_len=None, training=False):
    assert isinstance(input, dict)
    x = input['content'] 

    batch_size = melt.get_shape(x, 0)
    if c_len is None or max_c_len is None:
      c_len, max_c_len = melt.length2(x)

    if self.rnn_no_padding:
      logging.info('------------------no padding! train or eval')
      c_len = max_c_len

    x = self.embedding(x)

    if FLAGS.use_char:
      cx = input['char']

      cx = tf.reshape(cx, [batch_size * max_c_len, FLAGS.char_limit])
      chars_len = melt.length(cx)
      cx = self.char_embedding(cx)
      cx = self.char_encode(cx, chars_len, training=training)
      cx = self.char_pooling(cx, chars_len)
      cx = tf.reshape(cx, [batch_size, max_c_len, 2 * self.num_units])

      if self.char_combiner == 'concat':
        x = tf.concat([x, cx], axis=2)
      elif self.char_combiner == 'sfu':
        x = self.char_sfu_combine(x, cx, training=training)

    if FLAGS.use_pos:
      px = input['pos']      
      px = self.pos_embedding(px)
      x = tf.concat([x, px], axis=2)

    if FLAGS.use_ner:
      nx = input['ner']      
      nx = self.ner_embedding(nx)
      x = tf.concat([x, nx], axis=2)

    x = self.encode(x, c_len, training=training)

    return x

from third.bert import modeling
class BertEncoder(melt.Model):
 #embedding not used just for compatct with TextEncoder
  def __init__(self, embedding=None):
    super(BertEncoder, self).__init__(embedding)

    self.init_checkpoint = None
    
    if FLAGS.bert_dir:
      bert_dir = FLAGS.bert_dir
      bert_config_file = f'{bert_dir}/bert_config.json' 
      bert_config = modeling.BertConfig.from_json_file(bert_config_file)
      self.init_checkpoint= f'{bert_dir}/bert_model.ckpt' 
    elif FLAGS.bert_config_file:
      bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    else:
      bert_config = {
        "attention_probs_dropout_prob": 0.1, 
        "directionality": "bidi", 
        "hidden_act": "gelu", 
        "hidden_dropout_prob": 0.1, 
        "hidden_size": 768, 
        "initializer_range": 0.02, 
        "intermediate_size": 3072, 
        "max_position_embeddings": 512, 
        "num_attention_heads": 12, 
        "num_hidden_layers": 12, 
        "pooler_fc_size": 768, 
        "pooler_num_attention_heads": 12, 
        "pooler_num_fc_layers": 3, 
        "pooler_size_per_head": 128, 
        "pooler_type": "first_token_transform", 
        "type_vocab_size": 2, 
        "vocab_size": gezi.Vocabulary(FLAGS.vocab).size()
      }
      bert_config = modeling.BertConfig.from_dict(bert_config)

    self.bert_config = bert_config

  def restore(self):
    tvars = tf.trainable_variables()
    (assignment_map,
    initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(
        tvars, self.init_checkpoint)

    tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

    logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)  

  def call(self, input, c_len=None, max_c_len=None, training=False):
    self.step += 1
    x = input['content'] if isinstance(input, dict) else input
    batch_size = melt.get_shape(x, 0) 
    model = modeling.BertModel(
      config=self.bert_config,
      is_training=training,
      input_ids=x,
      input_mask=(x > 0) if c_len is not None else None)

    if self.step == 0 and self.init_checkpoint:
      self.restore()
    x = model.get_sequence_output()
    return x
