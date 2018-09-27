#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2018-09-17 20:34:23.281520
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

# TODO add support for na or not binary sigmoid loss
def criterion(model, x, y, training=False):
  y_ = model(x, training=training)
  if not FLAGS.type1_weight:
    weights = 1.0  
  else:
    # weights not help seems
    weights = tf.map_fn(lambda x: tf.cond(tf.equal(x, 1), lambda: FLAGS.type1_weight, lambda: 1.), x['type'], dtype=tf.float32)

  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_, weights=weights) 

