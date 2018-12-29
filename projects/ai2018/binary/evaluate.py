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

flags.DEFINE_string('info_path', None, '')
flags.DEFINE_bool('auc_need_softmax', True, '')

flags.DEFINE_string('class_weights_path', './mount/temp/ai2018/sentiment/class_weights.npy', '')
flags.DEFINE_float('logits_factor', 10, '10 7239 9 7245 but test set 72589 and 72532 so.. a bit dangerous')

flags.DEFINE_bool('show_detail', False, '')

flags.DEFINE_string('i', '.', '')

flags.DEFINE_string('metric_name', 'adjusted_f1/mean', '')
flags.DEFINE_float('min_thre', 0., '0.705')
flags.DEFINE_integer('len_thre', 256, '')
flags.DEFINE_float('max_thre', 1000., '')
flags.DEFINE_bool('adjust', True, '')
flags.DEFINE_bool('more_adjust', True, '')

#from sklearn.utils.extmath import softmax
from sklearn.metrics import log_loss, roc_auc_score

import numpy as np
import glob
import gezi
import melt 
logging = melt.logging

from wenzheng.utils import ids2text


import pickle
import pandas as pd
import traceback

#since we have same ids... must use valid and test 2 infos
valid_infos = {}
test_infos = {}

decay = None
wnames = []

def init():
  global valid_infos, test_infos
  
  with open(FLAGS.info_path, 'rb') as f:
    valid_infos = pickle.load(f)
  if FLAGS.test_input:
    with open(FLAGS.info_path.replace('.pkl', '.test.pkl'), 'rb') as f:
      test_infos = pickle.load(f)

  ids2text.init()

def calc_loss(labels, predicts):
  """
  softmax loss, mean loss and per attr loss
  """
  names = ['loss']
  loss = log_loss(labels, predicts)
  vals = [loss] 

  return vals, names

# TODO understand macro micro..
def calc_auc(labels, predicts):
  """
  per attr auc
  """
  names = ['auc']
  auc = roc_auc_score(labels, predicts)
  vals = [auc] 

  return vals, names

def evaluate(labels, predicts):
  vals, names = calc_auc(labels, predicts)
  vals_loss, names_loss = calc_loss(labels, predicts)
  
  vals += vals_loss
  names += names_loss

  return vals, names
  
