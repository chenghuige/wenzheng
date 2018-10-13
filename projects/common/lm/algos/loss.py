#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2018-10-11 22:14:50.914666
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
  
def loss_fn(model, inputs, targets, training=False):
  #targets = tf.transpose(targets, [1,0])
  labels = tf.reshape(targets, [-1])
  outputs = model(inputs, training=training)
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=outputs))
