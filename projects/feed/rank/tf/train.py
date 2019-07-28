#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read.py
#        \author   chenghuige  
#          \date   2019-07-26 18:02:22.038876
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf

from dataset import *
from model import *
import model as base
import evaluate as ev

keras = tf.keras
from keras import backend as K

def binary_crossentropy_with_ranking(y_true, y_pred):
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    logloss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
    
    # next, build a rank loss
    
    # clip the probabilities to keep stability
    y_pred_clipped = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    # translate into the raw scores before the logit
    y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))

    # determine what the maximum score for a zero outcome is
    y_pred_score_zerooutcome_max = K.max(y_pred_score * (y_true <1))

    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max

    # only keep losses for positive outcomes
    rankloss = rankloss * y_true

    # only keep losses where the score is below the max
    rankloss = K.square(K.clip(rankloss, -100, 0))

    # average the loss for just the positive outcomes
    rankloss = K.sum(rankloss, axis=-1) / (K.sum(y_true > 0) + 1)

    # return (rankloss + 1) * logloss - an alternative to try
    return rankloss + logloss

def main(_):
  melt.apps.init()
  fit = melt.apps.get_fit()
  FLAGS.eval_batch_size = 512 * 100
  model_name = FLAGS.model
  model = getattr(base, model_name)() 
  Dataset = TextDataset if not 'tfrecord' in FLAGS.train_input else TFRecordDataset
  loss_fn = tf.losses.sigmoid_cross_entropy if not FLAGS.rank_loss else binary_crossentropy_with_ranking
  print('--------------', model, Dataset, loss_fn)
  fit(Dataset,
      model,  
      tf.losses.sigmoid_cross_entropy,
      eval_fn=ev.evaluate,
      valid_write_fn=ev.valid_write)   

if __name__ == '__main__':
  tf.app.run()  
