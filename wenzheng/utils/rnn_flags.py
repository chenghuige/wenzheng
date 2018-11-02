#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn_flags.py
#        \author   chenghuige  
#          \date   2016-12-24 17:02:28.330058
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS
 
flags.DEFINE_string('rnn_method', 'forward', '0 forward, 1 backward, 2 bidirectional')
flags.DEFINE_string('rnn_output_method', 'max', '0 sumed vec, 1 last vector, 2 first vector, 3 all here first means first to original sequence')

flags.DEFINE_string('cell', 'gru', 'might set to lstm_block which is faster, or lstm_block_fused, cudnn_lstm even faster')
flags.DEFINE_string('cudnn_cell', None, 'gru or lstm')
flags.DEFINE_string('encoder_cell',  None, 'might set to lstm_block which is faster, or lstm_block_fused, cudnn_lstm even faster')
flags.DEFINE_string('decoder_cell', None, 'might set to lstm_block which is faster, or lstm_block_fused, cudnn_lstm even faster')
flags.DEFINE_integer('num_layers', 1, 'or > 1')
flags.DEFINE_integer('encoder_num_layers', None, 'or > 1')
flags.DEFINE_float('encoder_keep_prob', None, '')
flags.DEFINE_integer('decoder_num_layers', None, 'or > 1')
flags.DEFINE_float('decoder_keep_prob', None, '')

flags.DEFINE_boolean('feed_initial_sate', False, """set true just like ptb_word_lm to feed 
                                                  last batch final state to be inital state 
                                                  but experiments not show better result(similar)""")
flags.DEFINE_integer('rnn_hidden_size', 512, 'rnn cell state hidden size, follow im2txt set default as 512')
flags.DEFINE_integer('encoder_rnn_hidden_size', None, '')
flags.DEFINE_integer('decoder_rnn_hidden_size', None, '')

flags.DEFINE_bool('recurrent_dropout', False, '')
flags.DEFINE_bool('bw_dropout', False, '')

flags.DEFINE_bool('rnn_train_init_state', True, '')
flags.DEFINE_bool('rnn_padding', False, 'if True padding when train, eval always padding')
flags.DEFINE_bool('rnn_no_padding', False, 'if True always no padding, train or eval')
flags.DEFINE_bool('pooling_no_padding', False, 'if True always not consider padding when pooling')
flags.DEFINE_bool('encoder_residual_connect', False, '')

# for pytorch..
flags.DEFINE_bool('torch_cudnn_rnn', False, 'pytorch using CudnnRnn or StackRnn')

flags.DEFINE_bool('concat_layers', True, 'by default concat layers as hkust rnet did')
