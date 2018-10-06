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
flags.DEFINE_bool('to_simplify', False, '')

import sys 
import os

import gezi


def filter(x):
  x = gezi.filter_quota(x).replace('\r', '\x01').replace('\n', '\x02').replace('<R>', '\x01').replace('<N>', '\x02').replace('\t', ' ')
  # simplify seems not help but might help diversity
  if FLAGS.to_simplify:
    x = gezi.to_simplify(x)
  # TODO if needed try to find case usefull or not　I think especally for sentiment not reading, lower is ok not loose important info like NIKE
  x = x.lower()
  return x
