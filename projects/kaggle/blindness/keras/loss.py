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

from keras.losses import binary_crossentropy, categorical_crossentropy

def earth_mover_loss(y_true, y_pred):
  cdf_ytrue = K.cumsum(y_true, axis=-1)
  cdf_ypred = K.cumsum(y_pred, axis=-1)
  samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
  return samplewise_emd

# reference link: https://www.kaggle.com/christofhenkel/weighted-kappa-loss-for-keras-tensorflow
#.. can no simple load
def kappa_loss(y_true, y_pred, y_pow=2, eps=1e-12, N=5, bsize=32, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)
    
        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))
    
        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)
    
        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)
    
        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.to_float(bsize))
    
        return nom*0.5 / (denom + eps) + categorical_crossentropy(y_true, y_pred)*0.5

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
    # classification
    if 'kappa' in loss_type:
      return kappa_loss
    return 'categorical_crossentropy'


