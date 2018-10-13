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
flags = tf.app.flags
FLAGS = flags.FLAGS

import melt 
logging = melt.logging

import inspect

def grad(model, x, y, loss_fn):
  with tf.GradientTape() as tape:
    if 'training' in inspect.getargspec(loss_fn).args:
      loss = loss_fn(model, x, y, training=True)
    else:
      loss = loss_fn(model, x, y)

  return loss, tape.gradient(loss, model.trainable_variables)

def clip_gradients(grads_and_vars, clip_ratio):
  gradients, variables = zip(*grads_and_vars)
  clipped, _ = tf.clip_by_global_norm(gradients, clip_ratio)
  return zip(clipped, variables)

def restore(model, ckpt_dir=None):
  if not ckpt_dir:
    ckpt_dir = FLAGS.model_dir + '/ckpt'
  
  if os.path.exists(ckpt_dir + '.index'):
    latest_checkpoint = ckpt_dir
  else:
    latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)

  logging.info('Latest checkpoint:', latest_checkpoint)

  checkpoint = tf.train.Checkpoint(model=model)      
  
  # TODO check return value, verify if it is restore ok ?
  checkpoint.restore(latest_checkpoint)
