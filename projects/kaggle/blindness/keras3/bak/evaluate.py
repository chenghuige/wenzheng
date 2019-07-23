#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2019-07-21 21:52:01.775723
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from keras.callbacks import Callback

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score

from config import *

class QWKEvaluation(Callback):
  def __init__(self, validation_data=(), batch_size=64, interval=1):
    super(Callback, self).__init__()

    self.interval = interval
    self.batch_size = batch_size
    self.valid_generator, self.y_val = validation_data
    self.history = []

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.interval == 0:
      y_pred = self.model.predict_generator(generator=self.valid_generator,
                                            steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),
                                            workers=1, use_multiprocessing=False,
                                            verbose=1)
      def flatten(y):
        return np.argmax(y, axis=1).reshape(-1)
      
      score = cohen_kappa_score(flatten(self.y_val),
                                flatten(y_pred),
                                labels=[0,1,2,3,4],
                                weights='quadratic')
      print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))
      self.history.append(score)
      if score >= max(self.history):
        print('saving checkpoint: ', score)
        self.model.save('../working/densenet_bestqwk.h5')

