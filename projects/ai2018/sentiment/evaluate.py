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
flags.DEFINE_bool('use_class_weights', False, '')

#from sklearn.utils.extmath import softmax
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from melt.utils.weight_decay import WeightDecay, WeightsDecay

import numpy as np
import gezi
import melt 
logging = melt.logging

from wenzheng.utils import ids2text
#from projects.ai2018.sentiment.algos.config import ATTRIBUTES, NUM_ATTRIBUTES, NUM_CLASSES, CLASSES
from algos.config import ATTRIBUTES, NUM_ATTRIBUTES, NUM_CLASSES, CLASSES


import pickle
import pandas as pd

#since we have same ids... must use valid and test 2 infos
valid_infos = {}
test_infos = {}

decay = None
wnames = []
classes = ['0na', '1neg', '2neu', '3pos']
num_classes = len(classes)

class_weights = None

def init():
  global valid_infos, test_infos
  global wnames
  global class_weights
  class_weights = np.load('/home/gezi/temp/ai2018/sentiment/class_weights.npy')

  with open(FLAGS.info_path, 'rb') as f:
    valid_infos = pickle.load(f)
  with open(FLAGS.info_path.replace('.pkl', '.test.pkl'), 'rb') as f:
    test_infos = pickle.load(f)

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


def regression_to_class(predict):
  if predict > 7:
    return 3
  elif predict > 5:
    return 2
  elif predict > 3:
    return 1
  else:
    return 0

def to_class(predicts, thre=0.5):
  if FLAGS.loss_type == 'hier':
    ## TODO even hier.. still not good below...
    # result = np.zeros([len(predicts)], dtype=np.int32)
    # for i, predict in enumerate(predicts):
    #   na_prob = gezi.sigmoid(predict[0])
    #   if na_prob > thre:
    #     result[i] = 0
    #   else:
    #     result[i] = np.argmax(predict[1:]) + 1
    # return result
    return np.argmax(predicts, -1)
  elif FLAGS.loss_type == 'regression':
    return np.array([regression_to_class(x) for x in predicts])
  else:
    return np.argmax(predicts, -1)
  
def calc_f1(labels, predicts, model_path=None):
  names = ['mean'] + ATTRIBUTES + classes
  num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
  if FLAGS.binary_class_index is not None:
    names = ['mean'] + ATTRIBUTES + ['not', classes[FLAGS.binary_class_index]] 

  names = ['f1/' + x for x in names]
  # TODO show all 20 * 4 ? not only show 20 f1
  f1_list = []
  class_f1 = np.zeros([num_classes])
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
    class_f1 += scores
    all_f1 += list(scores)
    f1 = np.mean(scores)
    f1_list.append(f1)
  f1 = np.mean(f1_list)
  class_f1 /= NUM_ATTRIBUTES

  vals = [f1] + f1_list + list(class_f1)

  if model_path is None:
    if FLAGS.decay_target and 'f1' in FLAGS.decay_target:
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

def calc_loss(labels, predicts, model_path=None):
  """
  softmax loss, mean loss and per attr loss
  """
  names = ['mean'] + ATTRIBUTES
  names = ['loss/' + x for x in names]
  losses = []
  for i in range(NUM_ATTRIBUTES):
    loss = log_loss(labels[:,i], predicts[:,i])
    losses.append(loss)
  vals = [np.mean(losses)] + losses

  if model_path is None:
    if FLAGS.decay_target and 'loss' in FLAGS.decay_target:
      if  FLAGS.num_learning_rate_weights <= 1:
        target = loss
      elif FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
        target = losses
      else:
        raise f'Unsupported weights number{FLAGS.num_learning_rate_weights}'

      weights = decay.add(target)
      if FLAGS.num_learning_rate_weights > 1:
        vals += list(weights)
        names += [f'weights/{name}' for name in wnames]

  return vals, names

# TODO understand macro micro..
def calc_auc(labels, predicts, model_path=None):
  """
  per attr auc
  """
  names = ['mean'] + ATTRIBUTES + classes
  names = ['auc/' + x for x in names]
  aucs_list = []
  class_aucs = np.zeros([num_classes])
  for i in range(NUM_ATTRIBUTES):
    aucs = []
    #print(np.sum(predicts[:,i], -1))
    for j in range(NUM_CLASSES):
      auc = roc_auc_score((labels[:, i] == j).astype(int), predicts[:,i, j])
      aucs.append(auc)
    auc = np.mean(aucs) 
    aucs_list.append(auc)
    class_aucs += np.array(aucs)
  class_aucs /= NUM_ATTRIBUTES
  auc = np.mean(aucs_list)
  vals = [auc] + aucs_list + list(class_aucs)

  if model_path is None:
    if FLAGS.decay_target and 'auc' in FLAGS.decay_target:
      if  FLAGS.num_learning_rate_weights <= 1:
        target = auc
      elif FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
        target = aucs_list
      else:
        raise f'Unsupported weights number{FLAGS.num_learning_rate_weights}'

      weights = decay.add(target)
      if FLAGS.num_learning_rate_weights > 1:
        vals += list(weights)
        names += [f'weights/{name}' for name in wnames]

  return vals, names

def evaluate(labels, predicts, ids=None, model_path=None):
  # TODO here use softmax will cause problem... not correct.. for f1
  probs = gezi.softmax(predicts)
  if FLAGS.use_class_weights:
    probs *= class_weights

  vals, names = calc_f1(labels, probs, model_path)
  
  vals_loss, names_loss = calc_loss(labels, probs, model_path)
  vals += vals_loss 
  names += names_loss
  
  probs = predicts if not FLAGS.auc_need_softmax else probs
  vals_auc, names_auc = calc_auc(labels, probs, model_path)
  vals += vals_auc 
  names += names_auc 

  return vals, names
  
valid_write = None
infer_write = None 

def write(ids, labels, predicts, ofile, ofile2=None, is_infer=False):
  infos = valid_infos if not is_infer else test_infos
  df = pd.DataFrame()
  df['id'] = ids
  contents = [infos[id]['content_str'] for id in ids]
  df['content'] = contents
  if labels is not None:
    for i in range(len(ATTRIBUTES)):
      df[ATTRIBUTES[i] + '_y'] = labels[:,i] - 2
  for i in range(len(ATTRIBUTES)):
    # nitice if na only then if -1 means predict na as finally should be -2
    if FLAGS.loss_type == 'regression':
      df[ATTRIBUTES[i]] = predicts[:,i]
    else:
      df[ATTRIBUTES[i]] = np.argmax(predicts[:,i], 1) - 2
  if is_infer:
    df.to_csv(ofile, index=False, encoding="utf_8_sig")
  num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 1
  if FLAGS.loss_type == 'regression':
    num_classes = 1
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

if __name__ == '__main__':
  class_weights = np.load('/home/gezi/temp/ai2018/sentiment/class_weights.npy')

  df = pd.read_csv(sys.argv[1])

  ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
                'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
                'price_level', 'price_cost_effective', 'price_discount', 
                'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
                'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
                'others_overall_experience', 'others_willing_to_consume_again']
  def parse(l):
    if ',' in l:
      # this is list save (list of list)
      return np.array([float(x.strip()) for x in l[1:-1].split(',')])
    else:
      # this numpy save (list of numpy array)
      return np.array([float(x.strip()) for x in l[1:-1].split(' ') if x.strip()])

  #scores = df['score_logits']
  scores = df['score']
  scores = [parse(score) for score in scores] 
  scores = np.array(scores)
  
  predicts = np.reshape(scores, [-1, NUM_ATTRIBUTES, NUM_CLASSES])  
  
  # for auc might need to do this 
  #predicts /= 26
  
  idx = 2
  length = NUM_ATTRIBUTES 

  labels = df.iloc[:,idx:idx+length].values
  labels += 2

  vals, names = evaluate(labels, predicts)

  for name, val in zip(names, vals):
    print(name, val)

  print('---------------------------------')
  for name, val in zip(names, vals):
    if 'mean' in name:
      print(name, val)