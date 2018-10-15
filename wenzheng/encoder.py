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

import sys 
import os

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

    logging.info(f'encoder:{type}')
    logging.info('encoder recurrent dropout:{}'.format(self.recurrent_dropout))
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
                                      bw_dropout=self.bw_dropout,
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