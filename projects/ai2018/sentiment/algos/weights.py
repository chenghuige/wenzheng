#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   weights.py
#        \author   chenghuige  
#          \date   2018-09-17 19:45:07.901500
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import sys 
import os
import numpy as np

from algos.config import NUM_CLASSES, NUM_ATTRIBUTES, ATTRIBUTES, ATTRIBUTES_MAP

import melt
logging = melt.logging

def no_weights():
  return FLAGS.aspect == 'all' and FLAGS.attr_index is None

def get_pos(aspect):
  start = None 
  end = None
  index = None
  if '-' in aspect:
    aspect, index = aspect.split('-')
    index = int(index)
  if aspect == 'location':
    start = 0
    end = 3
  elif aspect == 'service':
    start = 3
    end = 7
  elif aspect == 'price':
    start = 7
    end = 10
  elif aspect == 'environment':
    start = 10
    end = 14
  elif aspect == 'dish':
    start = 14
    end = 18
  elif aspect == 'others':
    start = 18
    end = 20 

  if index is not None:
    start += index 
    end = start + 1

  return start, end

def parse_weights():
  if ':' in FLAGS.weights:
    weights = np.ones([NUM_ATTRIBUTES]) * FLAGS.init_weight
    for item in FLAGS.weights.split(','):
      aspect, val = item.split(':')
      val = float(val)
      start, end = get_pos(aspect)
      if start is None:
        assert aspect in ATTRIBUTES_MAP
        weights[ATTRIBUTES_MAP[aspect]] = val
      else:
        for i in range(start, end):
          weights[i] = val
  else:
    weights = list(map(float, FLAGS.weights.split(',')))
    assert len(weights) == NUM_ATTRIBUTES

  return weights

def get_weights(aspect, attr_index=None):
  if FLAGS.weights:
    weights = parse_weights()
  else:
    weights = np.zeros([NUM_ATTRIBUTES])

    if no_weights():
      return 1.
    
    if attr_index is not None:
      weights[attr_index] = 1. 

    start, end = get_pos(aspect)
    if start is None:
      assert aspect in ATTRIBUTES_MAP
      weights[ATTRIBUTES_MAP[aspect]] = 1.
    else:
      for i in range(start, end):
        weights[i] = 1.

  logging.info('weights:', weights)
  logging.info('weights:', list(zip(ATTRIBUTES, weights)))
  return tf.expand_dims(tf.constant(weights, dtype=tf.float32), 0)
