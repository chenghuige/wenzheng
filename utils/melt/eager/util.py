#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2018-09-03 12:05:03.098236
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf

def grad(model, x, y, loss_fn):
  with tf.GradientTape() as tape:
    try:
      loss = loss_fn(model, x, y, training=True)
    except Exception:
      loss = loss_fn(model, x, y)
  return loss, tape.gradient(loss, model.trainable_variables)

def clip_gradients(grads_and_vars, clip_ratio):
  gradients, variables = zip(*grads_and_vars)
  clipped, _ = tf.clip_by_global_norm(gradients, clip_ratio)
  return zip(clipped, variables)