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
from sklearn.metrics import f1_score

from melt.utils.weight_decay import WeightDecay

import numpy as np
import gezi
import melt 
logging = melt.logging

from wenzheng.utils import ids2text
from algos.config import ATTRIBUTES

import pickle
import pandas as pd

infos = {}
decay = None

def init():
  global infos 
  with open(FLAGS.info_path, 'rb') as f:
    infos = pickle.load(f)

  ids2text.init()

def calc_f1(labels, predicts, ids, model_path):
  names = ['mean'] + ATTRIBUTES
  names = ['f1/' + x for x in names]
  # TODO show all 20 * 4 ? not only show 20 f1
  f1_list = []
  for i in range(len(ATTRIBUTES)):
    f1 = f1_score(labels[:,i], np.argmax(predicts[:,i], 1) - 2, average='macro')
    f1_list.append(f1)
  f1 = np.mean(f1_list)

  if model_path is None:
    if tf.executing_eagerly():
      logging.info('eager mode not support decay right now')
    elif FLAGS.decay_target:
      global decay
      decay_target = FLAGS.decay_target
      if not decay:
        logging.info('decay_target:', decay_target)
        cmp = 'less' if decay_target == 'loss' else 'greater'
        decay = WeightDecay(patience=FLAGS.decay_patience, 
                            decay=FLAGS.decay_factor, 
                            cmp=cmp,
                            min_weight=0.00001)
      decay.add(f1)
  
  return [f1] + f1_list, names
  
valid_write = None
infer_write = None 

def write(ids, labels, predicts, ofile, ofile2=None, is_infer=False):
  df = pd.DataFrame()
  df['id'] = ids
  contents = [infos[id]['content_str'] for id in ids]
  df['content'] = contents
  if labels is not None:
    for i in range(len(ATTRIBUTES)):
      df[ATTRIBUTES[i] + '_y'] = labels[:,i]
  for i in range(len(ATTRIBUTES)):
    df[ATTRIBUTES[i]] = np.argmax(predicts[:,i], 1) - 2
  df.to_csv(ofile, index=False, encoding="utf_8_sig")
  if is_infer:
    df2 = df
    df2['seg'] = [ids2text.ids2text(infos[id]['content'], sep='|') for id in ids]
    df2.to_csv(ofile2, index=False, encoding="utf_8_sig")

def valid_write(ids, labels, predicts, ofile):
  return write(ids, labels, predicts, ofile)

def infer_write(ids, predicts, ofile, ofile2):
  return write(ids, None, predicts, ofile, ofile2, is_infer=True)