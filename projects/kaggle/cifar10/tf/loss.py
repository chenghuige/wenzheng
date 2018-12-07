#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2018-12-07 15:19:21.172303
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf

def criterion(model, x, y, training=False):
  y_ = model(x, training=training)
  weight_decay = 0.0002

  loss = tf.losses.sparse_softmax_cross_entropy(
      logits=y_, labels=y)
  loss = tf.reduce_mean(loss)

  model_params = tf.trainable_variables()
  loss += weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in model_params])
  return loss  

