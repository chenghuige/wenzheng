#!/usr/bin/env python
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige  
#          \date   2016-07-19 17:09:07.466651
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, time
import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

from algos.config import CLASSES
from sklearn.metrics import roc_auc_score

def calc_auc(labels, predicts):
  total_auc = 0. 
  aucs = [0.] * len(CLASSES)
  for i, class_ in enumerate(CLASSES):
    auc = roc_auc_score(labels[:, i], predicts[:, i])
    aucs[i] = auc
    total_auc += auc
  auc = total_auc / len(CLASSES) 
  vals = [auc] + aucs
  names = ['auc/avg'] + ['auc/%s' % x for x in CLASSES]
  return vals, names