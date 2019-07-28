#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2019-07-28 08:43:41.067128
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from sklearn.metrics import roc_auc_score, log_loss
import gezi

dataset = None 

def evaluate(y, y_, model_path=None):
  y_ = gezi.sigmoid(y_)
  auc = roc_auc_score(y, y_)
  loss = log_loss(y, y_)
  return [auc, loss], ['auc', 'loss']

def valid_write(ids, labels, predicts, out):
  for id, label, predict in zip(ids, labels, predicts):
    print('{},{},{:.3f}'.format(id, label, gezi.sigmoid(predict)), file=out)  
