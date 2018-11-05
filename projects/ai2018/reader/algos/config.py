#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2018-02-16 19:12:02.066189
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

CLASSES = ['neg', 'pos', 'na']
NUM_CLASSES = 3

flags.DEFINE_integer('char_limit', 6, '')
flags.DEFINE_bool('use_char', False, '')

flags.DEFINE_string('model', None, '')

flags.DEFINE_bool('use_type', False, '')
flags.DEFINE_bool('use_type_emb', False, '')
flags.DEFINE_bool('use_type_rnn', False, '')
flags.DEFINE_integer('type_emb_dim', 100, '')
flags.DEFINE_bool('rcontent', False, '')

flags.DEFINE_bool('split_type', False, '')

flags.DEFINE_string('att_type', 'p2q', '')
flags.DEFINE_string('att_combiner', 'gate', '')

flags.DEFINE_string('att_activation', 'relu', '')

flags.DEFINE_integer('unk_vocab_size', None, 'none means not to use, and if use can set like 10000')
flags.DEFINE_bool('finetune_unk_vocab', False, '')

flags.DEFINE_bool('type1_only', False, '')
flags.DEFINE_bool('type0_only', False, '')
flags.DEFINE_float('type1_weight', None, '')
flags.DEFINE_integer('type1_count', None, '')

flags.DEFINE_bool('use_qc_att', True, '')
flags.DEFINE_bool('use_bidaf_att', False, '')
flags.DEFINE_bool('use_ac_att', False, '')
flags.DEFINE_bool('use_self_match', True, '')

flags.DEFINE_bool('cq_reverse', False, '')

flags.DEFINE_bool('use_answer_emb', False, '')

flags.DEFINE_bool('combine_query', False, '')

flags.DEFINE_bool('share_dropout', True, '')

flags.DEFINE_bool('use_att_encode', False, 'for mreader wether to use at encode or not')

flags.DEFINE_bool('use_bias', False, '')

flags.DEFINE_string('encoder_combiner', 'concat', '')

flags.DEFINE_bool('use_mlp', False, '')
flags.DEFINE_bool('use_word', True, '')

flags.DEFINE_bool('use_pos', False, '')
flags.DEFINE_bool('use_tag', False, '')
flags.DEFINE_bool('use_ner', False, '')
flags.DEFINE_integer('tag_emb_dim', 100, '')

flags.DEFINE_bool('use_emb_fc', False, '')
flags.DEFINE_bool('use_emb_att', False, '')
flags.DEFINE_bool('use_emb_max', False, '')

flags.DEFINE_float('sfu_keepprob', 0.5, '')
flags.DEFINE_float('emb_keepprob', 0.5, '')

flags.DEFINE_bool('emb_dropout', False, '')

flags.DEFINE_bool('use_label_emb', False, '')
flags.DEFINE_bool('use_label_att', False, '')
flags.DEFINE_bool('simple_label_att', False, '')

flags.DEFINE_string('label_attention_combiner', 'gate', 'gate or dsfu')
flags.DEFINE_string('self_attention_combiner', 'gate', 'gate or dsfu')

flags.DEFINE_integer('label_emb_dim', 100, '')
flags.DEFINE_integer('label_emb_height', None, '')
flags.DEFINE_bool('concat_label_emb', False, '')
flags.DEFINE_bool('label2text_attention', False, '')
flags.DEFINE_bool('perlabel_encoding', False, '')

flags.DEFINE_bool('toxic_only', False, '')
flags.DEFINE_bool('toxic_softmax_loss', False, 'for toxic and serv')

flags.DEFINE_bool('char_only', False, '')
flags.DEFINE_integer('char_num_layers', 1, '')
flags.DEFINE_integer('char_hidden_size', 100, '')
flags.DEFINE_string('char_output_method', 'last', '')

flags.DEFINE_integer('simple_char_num_layers', 3, '')

flags.DEFINE_string('addtional_word_info', None, 'pos,tag,ner')

flags.DEFINE_string('decay_target', None, 'loss or auc')
flags.DEFINE_integer('decay_patience', 4, '')
flags.DEFINE_float('decay_factor', 0.5, '')

flags.DEFINE_bool('optimize_auc', False, '')
flags.DEFINE_float('auc_ratio', 0., '')
flags.DEFINE_bool('balance_pos_neg', False, '')

flags.DEFINE_bool('dynamic_weights', False, '')

flags.DEFINE_bool('use_gate', False, '')

flags.DEFINE_bool('hate_corpus', False, '')

flags.DEFINE_bool('use_position_encoding', False, '')
