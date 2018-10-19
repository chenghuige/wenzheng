#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   filter.py
#        \author   chenghuige  
#          \date   2018-09-14 21:55:52.069104
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('simplify', False, '')

import sys 
import os

import gezi


def filter_duplicate(text):
  return ''.join([x for i, x in enumerate(text) if not (i < len(text) - 1 and not(x.strip()) and not(text[i + 1].strip()))])

def filter(x):
  x = filter_duplicate(x)
  x = gezi.filter_quota(x).replace('\r', '\x01').replace('\n', '\x02').replace('<R>', '\x01').replace('<N>', '\x02').replace('\t', ' ').replace(' ', '\x03')
  # simplify seems not help but might help diversity
  if FLAGS.simplify:
    try:
      x = gezi.to_simplify(x)
    except Exception:
      pass
  # TODO if needed try to find case usefull or not　I think especally for sentiment not reading, lower is ok not loose important info like NIKE
  x = x.lower()
  return x
