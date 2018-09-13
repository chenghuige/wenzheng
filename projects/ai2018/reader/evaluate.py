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

#from sklearn.utils.extmath import softmax

from melt.utils.weight_decay import WeightDecay

import numpy as np
import gezi
import melt 
logging = melt.logging

from wenzheng.utils import ids2text

import pickle

infos = {}
def init():
  global infos 
  with open(FLAGS.info_path, 'rb') as f:
    infos = pickle.load(f)

  ids2text.init()

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

valid_names = ['id', 'label', 'predict', 'score', 'candidates', 'type', 'query', 'passage', 'query_seg', 'passage_seg']

def write(id, label, predict, out, out2=None, is_infer=False):
  info = infos[id]
  score = gezi.softmax(predict)
  predict = np.argmax(predict)
  candidates = info['candidates'].split('|')
  if label is not None:
    label = candidates[label]
  predict = candidates[predict]
  print(id, label, predict, score, candidates, info['type'], info['query_str'], info['passage_str'],
        ids2text.ids2text(info['query'], sep='|'), ids2text.ids2text(info['passage'], sep='|'), sep='\t', file=out)
  if is_infer:
    print(id, predict, sep='\t', file=out2)

def valid_write(id, label, predict, out):
  return write(id, label, predict, out)

def infer_write(id, predict, out, out_debug):
  return write(id, None, predict, out_debug, out, is_infer=True)