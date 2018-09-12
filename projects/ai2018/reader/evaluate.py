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

from tensorboard import summary as summary_lib

from collections import defaultdict

from melt.utils.weight_decay import WeightDecay

import numpy as np
import gezi
import melt 
logging = melt.logging

import pickle

infos = {}
def init():
  global infos 
  with open(FLAGS.info_path, 'rb') as f:
    infos = pickle.load(f)
  
decay = None

def calc_acc(labels, predicts, ids, model_path):
  names = ['acc', 'acc_if', 'acc_whether'] 
  predicts = np.argmax(predicts, 1)
  acc = np.mean(np.equal(labels, predicts))

  predicts1, predicts2, labels1, labels2 = [], [], [], []
  for i, id in enumerate(ids):
    type = infos[id]['type']
    if type == 0:
      predicts1.append(predicts[i])
      labels1.append(labels[i])
    else:
      predicts2.append(predicts[i])
      labels2.append(labels[i])
    
  acc_if = np.mean(np.equal(labels1, predicts1))
  acc_whether = np.mean(np.equal(labels2, predicts2))

  if model_path is None:
    if tf.executing_eagerly():
      logging.info('eager mode not support decay right now')
    else:
      global decay
      decay_target = FLAGS.decay_target
      if not decay:
        logging.info('decay_target:', decay_target)
        cmp = 'less' if decay_target == 'loss' else 'greater'
        decay = WeightDecay(patience=FLAGS.decay_patience, 
                            decay=FLAGS.decay_factor, 
                            cmp=cmp,
                            min_weight=0.00001)
      decay.add(acc)

  return [acc, acc_if, acc_whether], names
  
valid_write = None
infer_write = None 

def infer_write(id, predict, out):
  info = infos[id]
  predict = np.argmax(predict)
  candidates = info['candidates'].split('|')
  predict = candidates[predict]
  print(id, predict, sep='\t', file=out)

def valid_write(id, label, predict, out):
  info = infos[id]
  predict = np.argmax(predict)
  candidates = info['candidates'].split('|')
  label = candidates[label]
  predict = candidates[predict]
  print(id, label, predict, info['query_str'], info['passage_str'], sep=',', file=out)