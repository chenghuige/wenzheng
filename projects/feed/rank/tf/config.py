#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2019-07-26 23:14:46.546584
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_string('field_file_path', '../input/feat_fields.old', '')
flags.DEFINE_string('feat_file_path', '../input/feature_index', '')
flags.DEFINE_string('emb_activation', None, '')
flags.DEFINE_string('dense_activation', 'relu', '')
flags.DEFINE_integer('max_feat_len', 100, '')
flags.DEFINE_integer('hidden_size', 50, '')

flags.DEFINE_bool('wide_addval', True, '')
flags.DEFINE_bool('deep_addval', False, '')
flags.DEFINE_bool('deep_field', False, '')
flags.DEFINE_string('deep_wide_combine', 'concat', '')
flags.DEFINE_string('pooling', 'sum', '')

flags.DEFINE_bool('field_emb', False, '')
flags.DEFINE_bool('index_addone', True, '')

flags.DEFINE_bool('rank_loss', False, '')

flags.DEFINE_integer('feature_dict_size', 3200000, '')
flags.DEFINE_integer('field_dict_size', 100, '')
