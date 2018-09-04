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

flags.DEFINE_string('encoder_combiner', 'concat', '')

flags.DEFINE_bool('use_mlp', False, '')
flags.DEFINE_bool('use_word', True, '')
flags.DEFINE_bool('use_simple_char', False, '')
flags.DEFINE_string('simple_char_combiner', 'concat', '')
flags.DEFINE_bool('use_char', False, '')
flags.DEFINE_bool('use_simple_ngrams', False, '')
flags.DEFINE_string('char_combiner', 'concat', '')

flags.DEFINE_bool('use_ngrams', False, '')
flags.DEFINE_bool('use_fngrams', False, '')
flags.DEFINE_integer('ngram_emb_dim', 300, '')
flags.DEFINE_string('ngram_combiner', 'sum', 'sum or concat or dsfu')
flags.DEFINE_string('ngram_self_combiner', 'sum', 'sum or concat')
flags.DEFINE_bool('ngram_only', False, '')

flags.DEFINE_bool('use_token_info', False, '')
flags.DEFINE_bool('use_info_fc', False, '')
flags.DEFINE_string('token_info_combiner', 'concat', '')

flags.DEFINE_bool('use_comment_info', False, '')
flags.DEFINE_bool('use_comment_info_fc', False, '')
flags.DEFINE_string('comment_info_combiner', 'concat', '')
flags.DEFINE_bool('comment_info_lang_only', False, '')

flags.DEFINE_bool('cudnn_gru_encode', False, 'depreciated, for safe you could just use gru_baseline.py with auc 0.954 single model')

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
flags.DEFINE_string('char_output_method', 'max', '')

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


CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
CLASSES_COUNTS = [15294, 1595, 8449, 478, 7877, 1405]
# train set total 
TOTAL_COUNTS = 159571

NUM_CLASSES = len(CLASSES)
