#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2019-07-21 21:52:43.549005
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf 

# def earth_mover_loss(y_true, y_pred):
#   cdf_ytrue = tf.cumsum(y_true, axis=-1)
#   cdf_ypred = tf.cumsum(y_pred, axis=-1)
#   samplewise_emd = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(cdf_ytrue - cdf_ypred)), axis=-1))
#   return samplewise_emd

import keras.backend as K

def earth_mover_loss(y_true, y_pred):
  cdf_ytrue = K.cumsum(y_true, axis=-1)
  cdf_ypred = K.cumsum(y_pred, axis=-1)
  samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
  return samplewise_emd

def get_loss(loss_type=None):
  if 'regression' in loss_type:
    if 'sigmoid2' in loss_type:
      return 'binary_crossentropy'
    if not 'mae' in loss_type:
      return 'mse'
    else:
      return 'mae'
  elif 'ordinal' in loss_type:
    return 'binary_crossentropy'
  elif 'earth' in loss_type:
    return earth_mover_loss
  else:
    return 'categorical_crossentropy'


