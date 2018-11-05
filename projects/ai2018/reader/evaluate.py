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

import sys, os, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('info_path', None, '')

#from sklearn.utils.extmath import softmax

from melt.utils.weight_decay import WeightDecay, WeightsDecay

import numpy as np
import gezi
import melt 
logging = melt.logging

from wenzheng.utils import ids2text

from sklearn.metrics import log_loss

import pickle

valid_infos = {}
test_infos = {}

decay = None
wnames = []

def init():
  global valid_infos, test_infos
  global wnames
  with open(FLAGS.info_path, 'rb') as f:
    valid_infos = pickle.load(f)
  with open(FLAGS.info_path.replace('.pkl', '.test.pkl'), 'rb') as f:
    test_infos = pickle.load(f)

  ids2text.init()

  #min_learning_rate = 1e-5
  min_learning_rate = FLAGS.min_learning_rate
  logging.info('Min learning rate:', min_learning_rate)
  if FLAGS.decay_target:
    global decay
    decay_target = FLAGS.decay_target
    cmp = 'less' if decay_target == 'loss' else 'greater'
    if not decay:
      logging.info('Weight decay target:', decay_target)
      if FLAGS.num_learning_rate_weights <= 1:
        decay = WeightDecay(patience=FLAGS.decay_patience, 
                      decay=FLAGS.decay_factor, 
                      cmp=cmp,
                      #decay_start_epoch=FLAGS.decay_start_epoch,
                      decay_start_epoch=1,
                      min_learning_rate=min_learning_rate)
      else:
        wnames = ['if', 'whether']
        decay = WeightsDecay(patience=FLAGS.decay_patience, 
                      decay=FLAGS.decay_factor, 
                      cmp=cmp,
                      min_learning_rate=min_learning_rate,
                      names=wnames)  

## worse then just simply argmax
# def to_predict(logits):
#   predicts = np.zeros([len(logits)])
#   probs = gezi.softmax(logits, 1) 
#   for i, prob in enumerate(probs):
#     if prob[2] > 0.4:
#       predicts[i] = 2
#     else:
#       predicts[i] = np.argmax(prob[:-1])
#   return predicts

def calc_acc(labels, predicts, ids, model_path):
  names = ['acc', 'acc_if', 'acc_whether', 'acc_na'] 
  predicts = np.argmax(predicts, 1)
  #predicts = to_predict(predicts)
  acc = np.mean(np.equal(labels, predicts))
  # na index is set to 2
  acc_na = np.mean(np.equal(np.equal(labels, 2), np.equal(predicts, 2)))

  predicts1, predicts2, labels1, labels2 = [], [], [], []
  for i, id in enumerate(ids):
    type = valid_infos[id]['type']
    if type == 0:
      predicts1.append(predicts[i])
      labels1.append(labels[i])
    else:
      predicts2.append(predicts[i])
      labels2.append(labels[i])
    
  acc_if = np.mean(np.equal(labels1, predicts1))
  acc_whether = np.mean(np.equal(labels2, predicts2))

  vals = [acc, acc_if, acc_whether, acc_na]
  if model_path is None:
    if FLAGS.decay_target:
      target = acc if FLAGS.num_learning_rate_weights <= 1 else [acc_if, acc_whether]
      weights = decay.add(target)
      if FLAGS.num_learning_rate_weights > 1:
        vals += list(weights)
        names += [f'weights/{name}' for name in wnames]

  return vals, names

def calc_loss(labels, predicts, ids, model_path=None):
  """
  softmax loss, mean loss and per attr loss
  """
  names = ['loss', 'loss_if', 'loss_wether'] 

  predicts1, predicts2, labels1, labels2 = [], [], [], []
  for i, id in enumerate(ids):
    type = valid_infos[id]['type']
    if type == 0:
      predicts1.append(predicts[i])
      labels1.append(labels[i])
    else:
      predicts2.append(predicts[i])
      labels2.append(labels[i])

  loss = log_loss(labels, predicts)
  loss_if = log_loss(labels1, predicts1)
  loss_wether = log_loss(labels2, predicts2)
  vals = [loss, loss_if, loss_wether]

  if model_path is None:
    if FLAGS.decay_target and FLAGS.decay_target == 'loss':
      if  FLAGS.num_learning_rate_weights <= 1:
        target = loss
      elif FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
        target = losses
      else:
        raise f'Unsupported weights number{FLAGS.num_learning_rate_weights}'

      weights = decay.add(target)

  return vals, names

def evaluate(labels, predicts, ids=None, model_path=None):
  vals, names = calc_acc(labels, predicts, ids, model_path)
  probs = gezi.softmax(predicts)
  vals_loss, names_loss = calc_loss(labels, probs, ids, model_path)
  
  vals += vals_loss
  names += names_loss

  return vals, names
  
valid_write = None
infer_write = None 

valid_names = ['id', 'label', 'predict', 'score', 'candidates', 'type', 'query', 'passage', 'query_seg', 'passage_seg']

def write(id, label, predict, out, out2=None, is_infer=False):
  infos = valid_infos if not is_infer else test_infos
  info = infos[id]
  score = gezi.softmax(predict)
  predict = np.argmax(predict)
  candidates = info['candidates'].split('|')
  if label is not None:
    label = candidates[label]
  predict = candidates[predict]
  #.. on P40 not ok.. WHY ? FIXME
  # print(id, label, predict, score, gezi.csv(info['candidates']), info['type'], gezi.csv(info['query_str']), gezi.csv(info['passage_str']),
  #       gezi.csv(ids2text.ids2text(info['query'], sep='|')), gezi.csv(ids2text.ids2text(info['passage'], sep='|')), sep=',', file=out)
  # File "/home/slurm/job/tmp/job-58821/wenzheng/projects/ai2018/reader/evaluate.py", line 178, in write 
  # print(id, label, predict, score, gezi.csv(info['candidates']), info['type'], gezi.csv(info['query_str']), gezi.csv(info['passage_str']), sep=',', file=out) 
  # UnicodeEncodeError: 'latin-1' codec can't encode characters in position 0-1: ordinal not in range(256) 
  print(id, label, predict, score, gezi.csv(info['candidates']), info['type'], gezi.csv(info['query_str']), gezi.csv(info['passage_str']), sep=',', file=out)
  #print(id, label, predict, score, sep=',', file=out)
  if is_infer:
    #for contest
    print(id, predict, sep='\t', file=out2)

def valid_write(id, label, predict, out):
  return write(id, label, predict, out)

def infer_write(id, predict, out, out_debug):
  return write(id, None, predict, out_debug, out, is_infer=True)
