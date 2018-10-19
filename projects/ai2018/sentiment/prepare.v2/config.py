#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2018-10-17 14:44:21.024368
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('use_char', False, '')
flags.DEFINE_integer('char_limit', 6, '')
flags.DEFINE_integer('word_limit', 3000, '')
flags.DEFINE_bool('use_tag', False, '')
