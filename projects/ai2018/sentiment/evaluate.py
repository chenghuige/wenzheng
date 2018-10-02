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

from melt.utils.weight_decay import WeightDecay, WeightsDecay

import numpy as np
import gezi
import melt 
logging = melt.logging

from wenzheng.utils import ids2text
from algos.config import ATTRIBUTES, NUM_ATTRIBUTES, NUM_CLASSES, CLASSES

import pickle
import pandas as pd

infos = {}

decay = None
wnames = []


def init():
  global infos 
  global wnames
  with open(FLAGS.info_path, 'rb') as f:
    infos = pickle.load(f)

  ids2text.init()

  #min_learning_rate = 1e-5
  min_learning_rate = FLAGS.min_learning_rate
  if FLAGS.decay_target:
    global decay
    decay_target = FLAGS.decay_target
    cmp = 'less' if decay_target == 'loss' else 'greater'
    if FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES * NUM_CLASSES:
      for attr in ATTRIBUTES:
        for j, cs in enumerate(CLASSES):
          wnames.append(f'{attr}_{j}{cs}')
    elif FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
      wnames = ATTRIBUTES
    if not decay:
      logging.info('Weight decay target:', decay_target)
      if FLAGS.num_learning_rate_weights <= 1:
        decay = WeightDecay(patience=FLAGS.decay_patience, 
                      decay=FLAGS.decay_factor, 
                      cmp=cmp,
                      decay_start_epoch=1,
                      min_learning_rate=min_learning_rate)
      else:
        decay = WeightsDecay(patience=FLAGS.decay_patience, 
                      decay=FLAGS.decay_factor, 
                      cmp=cmp,
                      min_learning_rate=min_learning_rate,
                      names=wnames)  

# def to_predict(logits):
#   probs = gezi.softmax(logits, 1)
#   result = np.zeros([len(probs)], dtype=np.int32)
#   for i, prob in enumerate(probs):
#     # it can imporve but depends 0.6 or 0.7  or 0.8 ?
#     if prob[0] >= 0.6:
#       result[i] = -2
#     else:
#       result[i] = np.argmax(prob[1:]) - 1

#   return result

def to_class(predicts, thre=0.5):
  if FLAGS.loss_type != 'hier':
    return np.argmax(predicts, -1)
  else:
    result = np.zeros([len(predicts)], dtype=np.int32)
    for i, predict in enumerate(predicts):
      na_prob = gezi.sigmoid(predict[0])
      if na_prob > thre:
        result[i] = 0
      else:
        result[i] = np.argmax(predict[1:]) + 1
    return result

def calc_f1(labels, predicts, ids=None, model_path=None):
  classes = ['0na', '1neg', '2neu', '3pos']
  names = ['mean'] + ATTRIBUTES + classes
  num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
  if FLAGS.binary_class_index is not None:
    names = ['mean'] + ATTRIBUTES + ['not', classes[FLAGS.binary_class_index]] 

  names = ['f1/' + x for x in names]
  # TODO show all 20 * 4 ? not only show 20 f1
  f1_list = []
  attr_f1 = np.zeros([num_classes])
  all_f1 = []
  for i in range(NUM_ATTRIBUTES):
    #f1 = f1_score(labels[:,i], np.argmax(predicts[:,i], 1) - 2, average='macro')
    # print(labels[:,i])
    # print(predicts[:,i])
    # print(len(labels[:,i]))
    # print(len(predicts[:,i]))
    scores = f1_score(labels[:,i], to_class(predicts[:,i]), average=None)
    # if FLAGS.binary_class_index is not None:
    #   scores = [scores[1]]
    ## this will be a bit better imporve 0.001, I will just use when ensemble
    #scores = f1_score(labels[:,i], to_predict(predicts[:,i]), average=None)
    attr_f1 += scores
    all_f1 += list(scores)
    f1 = np.mean(scores)
    f1_list.append(f1)
  f1 = np.mean(f1_list)
  attr_f1 /= NUM_ATTRIBUTES
  
  vals = [f1] + f1_list + list(attr_f1)
  
  if model_path is None:
    if FLAGS.decay_target:
      if  FLAGS.num_learning_rate_weights <= 1:
        target = f1
      elif FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES * NUM_CLASSES:
        target = all_f1
      elif FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
        target = f1_list
      else:
        raise f'Unsupported weights number{FLAGS.num_learning_rate_weights}'
 
      weights = decay.add(target)
      if FLAGS.num_learning_rate_weights > 1:
        vals += list(weights)
        names += [f'weights/{name}' for name in wnames]
        
  return vals, names
  
valid_write = None
infer_write = None 

def write(ids, labels, predicts, ofile, ofile2=None, is_infer=False):
  df = pd.DataFrame()
  df['id'] = ids
  contents = [infos[id]['content_str'] for id in ids]
  df['content'] = contents
  if labels is not None:
    for i in range(len(ATTRIBUTES)):
      df[ATTRIBUTES[i] + '_y'] = labels[:,i] - 2
  for i in range(len(ATTRIBUTES)):
    # nitice if na only then if -1 means predict na as finally should be -2
    df[ATTRIBUTES[i]] = np.argmax(predicts[:,i], 1) - 2
  if is_infer:
    df.to_csv(ofile, index=False, encoding="utf_8_sig")
  num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
  df['score'] = [list(x) for x in np.reshape(predicts, [-1, NUM_ATTRIBUTES * num_classes])]
  if not is_infer:
    df['seg'] = [ids2text.ids2text(infos[id]['content'], sep='|') for id in ids]
    df.to_csv(ofile, index=False, encoding="utf_8_sig")
  if is_infer:
    df2 = df
    df2['seg'] = [ids2text.ids2text(infos[id]['content'], sep='|') for id in ids]
    df2.to_csv(ofile2, index=False, encoding="utf_8_sig")

def valid_write(ids, labels, predicts, ofile):
  return write(ids, labels, predicts, ofile)

def infer_write(ids, predicts, ofile, ofile2):
  return write(ids, None, predicts, ofile, ofile2, is_infer=True)