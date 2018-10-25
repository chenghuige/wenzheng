#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   text2ids.py
#        \author   chenghuige  
#          \date   2018-09-18 12:18:31.627281
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('feed_single_en', True, '')
flags.DEFINE_bool('to_lower', True, '')
flags.DEFINE_integer('unk_vocab_size', None, 'none means not to use, and if use can set like 10000')


import sys 
import os

#from projects.ai2018.reader.prepare import filter
from wenzheng.utils.text2ids import text2ids as to_ids
import wenzheng
  
def text2ids(text):
  wenzheng.utils.text2ids.init()
  #text = filter.filter(text)
  return to_ids(text, seg_method=FLAGS.seg_method, 
                feed_single_en=FLAGS.feed_single_en,
                to_lower=FLAGS.to_lower,
                norm_digit=False,
                pad=False,
                unk_vocab_size=FLAGS.unk_vocab_size)
