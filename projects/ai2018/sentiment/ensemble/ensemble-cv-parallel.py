#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ensemble.py
#        \author   chenghuige  
#          \date   2018-09-15 19:0num_classes:21.026718
#   \Description   ensemble by OOF score blend per attribute
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('debug', True, '')
flags.DEFINE_bool('grid_search', True, '')
flags.DEFINE_string('method', 'blend', '')
flags.DEFINE_string('idir', '.', '')
flags.DEFINE_float('norm_factor', 0.0001, 'attr weights used norm factor')
flags.DEFINE_float('logits_factor', 10, '10 7239 9 7245 but test set 72589 and 72532 so.. a bit dangerous')
flags.DEFINE_string('weight_by', 'adjusted_f1', '')
flags.DEFINE_integer('num_grids', 10, '')
flags.DEFINE_bool('adjust', True, '')
flags.DEFINE_bool('more_adjust', True, '')
flags.DEFINE_integer('seed', None, '')
flags.DEFINE_integer('num_folds', 5, '')
flags.DEFINE_bool('show_detail', False, '')

import sys 
import os

import glob
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import minmax_scale
import gezi
from tqdm import tqdm
import math
from scipy.stats import rankdata

from sklearn.model_selection import KFold

DEBUG = 0
idir = '.'

ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
              'price_level', 'price_cost_effective', 'price_discount', 
              'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
              'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again']

CLASSES = ['na', 'neg', 'neu', 'pos']

num_attrs = len(ATTRIBUTES)
NUM_ATTRIBUTES = num_attrs
num_classes = 4
NUM_CLASSES = num_classes

num_ensembles = 0

def parse(l):
  if ',' in l:
    # this is list save (list of list)
    return np.array([float(x.strip()) for x in l[1:-1].split(',')])
  else:
    # this numpy save (list of numpy array)
    return np.array([float(x.strip()) for x in l[1:-1].split(' ') if x.strip()])

def calc_f1(labels, predicts):
  f1_list = []
  for i in range(num_attrs):
    f1 = f1_score(labels[:,i], predicts[:, i], average='macro')
    f1_list.append(f1)
    #f1_scores[i].append(f1)
  f1 = np.mean(f1_list)
  return f1 

def calc_f1s(labels, predicts):
  f1_list = []
  for i in range(num_attrs):
    f1 = f1_score(labels[:,i], predicts[:, i], average='macro')
    f1_list.append(f1)
  return np.array(f1_list)

def calc_losses(labels, predicts):
  losses = []
  for i in range(NUM_ATTRIBUTES):
    loss = log_loss(labels[:,i], predicts[:,i])
    losses.append(loss)
  return np.array(losses)

def calc_loss(labels, predicts):
  losses = []
  for i in range(NUM_ATTRIBUTES):
    loss = log_loss(labels[:,i], predicts[:,i])
    losses.append(loss)
  return np.mean(losses)

def calc_aucs(labels, predicts):
  aucs_list = []
  for i in range(NUM_ATTRIBUTES):
    aucs = []
    #print(np.sum(predicts[:,i], -1))
    for j in range(NUM_CLASSES):
      auc = roc_auc_score((labels[:, i] == j).astype(int), predicts[:, i, j])
      aucs.append(auc)
    auc = np.mean(aucs) 
    aucs_list.append(auc)
  return np.array(aucs_list)

def calc_f1_alls(labels, predicts):
  f1_list = []
  class_f1 = np.zeros([num_classes])
  for i in range(NUM_ATTRIBUTES):
    scores = f1_score(labels[:,i], predicts[:,i], average=None)
    class_f1 += scores
    f1 = np.mean(scores)
    f1_list.append(f1)
  f1 = np.mean(f1_list)
  class_f1 /= NUM_ATTRIBUTES
  return f1, f1_list, class_f1

class_weights_path = './class_weights.npy'
if not os.path.exists(class_weights_path):
  class_weights_path = '/home/gezi/temp/ai2018/sentiment/class_weights.npy'
# class_weights is per class(na, neg, neu, pos) weight
class_weights = np.load(class_weights_path)
#print('class_weights', class_weights)

if FLAGS.adjust:
  for i in range(len(class_weights)):
    for j in range(4):
      #class_weights[i][j] = math.log(class_weights[i][j])
      #class_weights[i] = gezi.softmax(class_weights[i])
      #class_weights[i][j] +=  math.sqrt(class_weights[i][j])
      #class_weights[i][j] += 0.
      #class_weights[i][j] = math.sqrt(class_weights[i][j])
      x = class_weights[i][j]
      # If using prob adjust just set x for logits seems x^3 better
      #class_weights[i][j] = x 
      # well this make single model adjusted f1 improve by adding 100...
      #class_weights[i][j] = x * x * x + 100
      class_weights[i][j] = x * x * x 

    #if FLAGS.more_adjust:
    #  #this has been tested to be effective as for both fold 0 and 1 and different model combinations
    #  class_weights[1][-2] = class_weights[1][-2] * 1.2
    #  class_weights[-2][0] = class_weights[-2][0] * 1.2

  if FLAGS.more_adjust:
    #this has been tested to be effective as for both fold 0 and 1 and different model combinations
    # pow(1.2, 22) 55.2061438912436 
    class_weights[1][-2] = class_weights[1][-2] * pow(1.2, 22)
    ## 22.63
    #x = pow(1.2, 18)  
    ## * 22644.802257413307
    #class_weights[-2][0] = class_weights[-2][0] * x * x * x * 1.2
    class_weights[-2][0] = class_weights[-2][0] * 60000
      # for i in range(len(class_weights)):
      #   for j in range(4):
      #     class_weights[i][j] /= np.sum(class_weights[i])

    #class_weights = gezi.softmax(class_weights)
  else:
    class_weights = np.ones_like(class_weights)
print('class_weights\n', class_weights)
ori_class_weights = class_weights

def to_predict(logits, weights=None, is_single=False, adjust=True):
  logits = np.reshape(logits, [-1, num_attrs, num_classes])
  ## DO NOT divde !!
  if is_single:
    factor = FLAGS.logits_factor
  else:
    if weights is None:
      factor = 1.
    else:
      factor =  FLAGS.logits_factor / weights
  #print('factor:', factor)

  if adjust and FLAGS.adjust or FLAGS.grid_search:
    logits = logits * factor
    probs = gezi.softmax(logits, -1) 
    probs *= class_weights
  else:
    probs = logits

  probs = np.reshape(probs, [-1, num_classes])
  result = np.zeros([len(probs)], dtype=int)
  for i, prob in enumerate(probs):
    # # TODO try to calibrate to 0.5 ?
    # if prob[0] >= 0.6:
    #   result[i] = -2
    # else:
    #   result[i] = np.argmax(prob[1:]) - 1

    # this can also improve but not as good as per attr class weights adjust, can get 7183
    # TODO class_weights right now still not the best!
    #prob[0] *= 0.4
    result[i] = np.argmax(prob) - 2
  
  result = np.reshape(result, [-1, num_attrs])
  return result

def blend_weights(weights, norm_facotr):
  for i in range(num_attrs):
    #weights[:, i] = minmax_scale(weights[:, i])
    ws = weights[:, i]
    #min_ws = np.min(ws)
    #max_ws = np.max(ws)
    #gap = max_ws - min_ws
    #if gap > 0:
    #  for j in range(len(weights)):
    #    weights[j][i] = ((weights[j][i] - min_ws) / gap) + norm_facotr
    ranked = rankdata(ws)
    sum_rank = np.sum(ranked)
    for j in range(len(weights)):
      weights[j][i] = ranked[j] / sum_rank

def get_counts(probs):
  predicts = np.argmax(probs, 1)
  counts = np.zeros(4)
  for predict in predicts:
    counts[predict] += 1
  return counts

def adjust_probs(probs, labels):
  f1 = f1_score(labels[:, 1] + 2, np.argmax(probs[:, 1], 1), average='macro')
  print('location_distance_from_business_district', f1)
  probs[:, 1][-2] *= 10
  f1 = f1_score(labels[:, 1] + 2, np.argmax(probs[:, 1], 1), average='macro')
  print('location_distance_from_business_district', f1)
  f1 = f1_score(labels[:, -2] + 2, np.argmax(probs[:, -2], 1), average='macro')
  print('thers_overall_experience', f1)
  probs[:, -2][0] *= 100000
  f1 = f1_score(labels[:, -2] + 2, np.argmax(probs[:, -2], 1), average='macro')
  print('thers_overall_experience', f1)

# TODO FIXME pymp not work class factors seems not locked...
import pymp
#class_factors = pymp.shared.array((num_attrs, num_classes), dtype='float') + 1.
# class factors is per class dynamic adjust for class weights
from multiprocessing import Manager 

def grid_search_class_factors(probs, labels, weights, num_grids=10):
  manager = Manager() 
  class_factors_dict = manager.dict()
  with pymp.Parallel(12) as p:
    for i in tqdm(p.range(num_attrs), ascii=True):
    #for i in p.range(num_attrs):
      #p.print(i, ATTRIBUTES[i])
      #p.print('init counts:', get_counts(probs[:, i]))
      index = np.argsort(-np.array(weights[i]))
      def is_ok(factor):
        return np.sum(np.argsort(-factor) == index) == 4
      best = 0
      for a in tqdm(range(1,1 + num_grids), ascii=True):
      #for a in(range(1,1 + num_grids)):
        for b in range(1,1 + num_grids):
          for c in range(1,1 + num_grids):
            for d in range(1,1 + num_grids):
              factor = np.array([a, b, c, d], dtype=np.float)
              factor2 = factor * weights[i]
              if not is_ok(factor2):
                continue
              preds = probs[:, i] * factor2 
              f1 = f1_score(labels[:, i] + 2, np.argmax(preds, 1), average='macro')
              if f1 > best:
                #p.print('\n', ATTRIBUTES[i], factor, factor2, f1)
                best = f1
                #class_factors[i] = factor
                class_factors_dict[i] = factor
                #p.print('counts:', get_counts(probs[:, i] * factor))
                #p.print('class_factors', i, class_factors_dict[i])

  class_factors = np.ones([num_attrs, num_classes])
  for i in range(num_attrs):
    class_factors[i] = class_factors_dict[i]
  return class_factors

def main(_):
  print('METHOD:', FLAGS.method)
  print('Norm factor:', FLAGS.norm_factor) 

  if FLAGS.num_grids <= 1:
    FLAGS.grid_search = False

  # if FLAGS.grid_search:
  #   FLAGS.debug = False

  DEBUG = FLAGS.debug 
  idir = FLAGS.idir

  # first id, sencod content ..
  idx = 2

  valid_files = glob.glob(f'{idir}/*.valid.csv')
  valid_files = [x for x in valid_files if not 'ensemble' in x]
  
  if not DEBUG:
    print('VALID then INFER')
    infer_files = glob.glob(f'{idir}/*.infer.csv.debug')
  else:
    print('Debug mode INFER ill write result using valid ids, just for test')
    infer_files = glob.glob(f'{idir}/*.valid.csv') 
    infer_files = [x for x in infer_files if not 'ensemble' in x]

  print('num_ensembles', len(valid_files), 'num_infers', len(infer_files))    
  assert len(valid_files) == len(infer_files), infer_files

  global num_ensembles
  num_ensembles = len(valid_files)

  # need global ? even only read?
  global class_weights
  #print('-----------', class_weights)

  kf = KFold(n_splits=FLAGS.num_folds, shuffle=True, random_state=FLAGS.seed)

  print('loading all valid csv')
  dfs = []
  for file_ in tqdm(valid_files, ascii=True):
    df = pd.read_csv(file_)
    df = df.sort_values('id')
    dfs.append(df)
  
  logits_f1_list = []
  logits_adjusted_f1_list = []
  probs_f1_list = []
  probs_adjusted_f1_list = []
  grids_logits_adjusted_f1_list = []

  logits_predict_list = []
  logits_adjusted_predict_list = []
  probs_predict_list = []
  probs_adjusted_predict_list = []
  grids_logits_adjusted_predict_list = []

  labels_list = []
  results_list = []

  def split_train_valid(x):
    if FLAGS.num_folds == 1:
      return x, x 
    else:
      total = 15000
      assert total % FLAGS.num_folds == 0
      num_valid = int(total / FLAGS.num_folds) 
      num_train = total - num_valid
      return x[:num_train], x[num_train:]

  for fold, (train_index, valid_index) in enumerate(kf.split(dfs[0])):
    print('FOLD_%s---------------------------' % fold)
    print(train_index, valid_index)
    class_factors = np.ones([num_attrs, num_classes])
    class_weights = ori_class_weights
    # logits sum results
    results = None
    # prob sum results
    results2 = None

    weights = [] 
    scores_list = []

    for fid, df in enumerate(dfs):
      file_ = valid_files[fid]
      train = df.iloc[train_index]
      valid =  df.iloc[valid_index]
      #if fid == 0:      
      train_labels = train.iloc[:, idx:idx+num_attrs].values
      valid_labels = valid.iloc[:, idx:idx+num_attrs].values
      labels = np.concatenate([train_labels, valid_labels], 0)
      train_predicts = train.iloc[:, idx+num_attrs:idx+2*num_attrs].values
      valid_predicts = valid.iloc[:, idx+num_attrs:idx+2*num_attrs].values
      predicts = np.concatenate([train_predicts, valid_predicts], 0)
      train_scores = train['score']
      valid_scores = valid['score']
      scores = np.concatenate([train_scores, valid_scores], 0)
      scores = [parse(score) for score in scores] 
      scores = np.array(scores)
      scores_list.append(scores)     

      train_labels, valid_labels = split_train_valid(labels)
      train_predicts, valid_predicts = split_train_valid(predicts)
      train_scores, valid_scores = split_train_valid(scores)
  
      f1s = calc_f1s(train_labels, train_predicts)
      f1s_adjusted = calc_f1s(train_labels, to_predict(train_scores, is_single=True))

      train_probs = gezi.softmax(train_scores.reshape([-1, NUM_ATTRIBUTES, NUM_CLASSES]))
      aucs = calc_aucs(train_labels + 2, train_probs)
      losses = calc_losses(train_labels + 2, train_probs)

      f1 = np.mean(f1s)
      f1_adjusted = np.mean(f1s_adjusted)
      
      print('%-3d' % fid, '%-100s' % file_, '%.5f' % f1, '%.5f' % f1_adjusted, '%.5f' % np.mean(aucs), '%.5f' % np.mean(losses)) 
      
      if FLAGS.weight_by == 'loss':
        weight = np.reshape(1 / losses, [num_attrs, 1])
      elif FLAGS.weight_by == 'auc':
        weight = np.reshape(aucs, [num_attrs, 1])
      else:
        weight = np.reshape(f1s_adjusted, [num_attrs, 1])

      weights.append(weight) 

    weights = np.array(weights)
    scores_list = np.array(scores_list)

    blend_weights(weights, FLAGS.norm_factor)

    # if DEBUG:
    #   print(weights)
    print('-----------calc weight and score')
    for fid in tqdm(range(len(valid_files)), ascii=True):
      scores = scores_list[fid]
      if results is None:
        results = np.zeros([len(scores), num_attrs * num_classes])
        results2 = np.zeros([len(scores), num_attrs * num_classes])
      weight = weights[fid]
      if FLAGS.method == 'avg' or FLAGS.method == 'mean': 
        weight = 1.
      for i, score in enumerate(scores):
        score = np.reshape(score, [num_attrs, num_classes]) * weight
        score = np.reshape(score, [-1])
      
        results[i] += score

        # notice softmax([1,2]) = [0.26894142, 0.73105858] softmax([2,4]) = [0.11920292, 0.88079708]
        score = np.reshape(score, [num_attrs, num_classes])
        
        # this not work because *weight already..
        #score *= FLAGS.logits_factor
        
        score = gezi.softmax(score, -1)
        
        #score *= class_weights

        score = np.reshape(score, [-1])
        
        results2[i] += score 

    sum_weights = np.sum(weights, 0)

    train_results, valid_results = split_train_valid(results)
    train_results2, valid_results2 = split_train_valid(results2)

    print('-----------using prob ensemble')
    adjusted_predict_prob = to_predict(valid_results2, sum_weights, adjust=False)
    adjusted_f1_prob = calc_f1(valid_labels, adjusted_predict_prob)
    valid_results2 = np.reshape(valid_results2, [-1, num_attrs, num_classes]) 
    predicts2 = np.argmax(valid_results2, -1) - 2
    f1_prob = calc_f1(valid_labels, predicts2)

    probs_f1_list.append(f1_prob)
    probs_adjusted_f1_list.append(adjusted_f1_prob)
    
    probs_predict_list.append(predicts2)
    probs_adjusted_predict_list.append(adjusted_predict_prob)
    
    print('%-40s' % 'f1_prob:', '%.5f' % f1_prob)
    print('%-40s' % 'adjusted f1_prob:', '%.5f' % adjusted_f1_prob)

    # print('-----------detailed f1 infos (ensemble by prob)')
    # _, adjusted_f1_probs, class_f1s = calc_f1_alls(valid_labels, to_predict(results2[num_train:], sum_weights, adjust=False))

    # for i, attr in enumerate(ATTRIBUTES):
    #   print(attr, adjusted_f1_probs[i])
    # for i, cls in enumerate(CLASSES):
    #   print(cls, class_f1s[i])

    print('-----------using logits ensemble')
    adjusted_predict = to_predict(valid_results, sum_weights)
    adjusted_f1 = calc_f1(valid_labels, adjusted_predict)
    valid_results = np.reshape(valid_results, [-1, num_attrs, num_classes]) 
    predicts = np.argmax(valid_results, -1) - 2
    f1 = calc_f1(valid_labels, predicts)

    logits_f1_list.append(f1)
    logits_adjusted_f1_list.append(adjusted_f1)

    logits_predict_list.append(predicts)
    logits_adjusted_predict_list.append(adjusted_predict)

    results_list.append(valid_results)
    labels_list.append(valid_labels)
    
    print('%-40s' % 'f1:', '%.5f' % f1)
    print('%-40s' % 'adjusted f1:', '%.5f' % adjusted_f1)

    if FLAGS.show_detail:
      print('-----------detailed f1 infos (ensemble by logits)')
      _, adjusted_f1s, class_f1s = calc_f1_alls(valid_labels, to_predict(valid_results, sum_weights))
      for i, attr in enumerate(ATTRIBUTES):
        print('%-40s' % attr, '%.5f' % adjusted_f1s[i])
      for i, cls in enumerate(CLASSES):
        print('%-40s' % cls, '%.5f' % class_f1s[i])

    print('%-40s' % 'f1:', '%.5f' % f1)
    print('%-40s' % 'f1 prob:', '%.5f' % f1_prob)
    print('%-40s' % 'adjusted f1 prob:', '%.5f' % adjusted_f1_prob)
    print('%-40s' % 'adjusted f1:', '%.5f' % adjusted_f1)

    if FLAGS.grid_search:
      print('------------grid search num_grids', FLAGS.num_grids)
      class_factors = grid_search_class_factors(gezi.softmax(np.reshape(train_results, [-1, num_attrs, num_classes]) * (FLAGS.logits_factor / sum_weights)), train_labels, class_weights, num_grids=FLAGS.num_grids)
        
      if FLAGS.show_detail:
        print('class_factors1 with num_grids', FLAGS.num_grids)
        print(class_factors)

      # adjust class weights to get better result from grid search 
      class_weights = class_weights * class_factors

      adjusted_f1_before_grids = adjusted_f1
      print('after dynamic adjust class factors')
      adjusted_predict = to_predict(valid_results, sum_weights)
      adjusted_f1 = calc_f1(valid_labels, adjusted_predict)
      valid_results = np.reshape(valid_results, [-1, num_attrs, num_classes]) 

      grids_logits_adjusted_f1_list.append(adjusted_f1)  
      grids_logits_adjusted_predict_list.append(adjusted_predict)

      print('-----------using logits ensemble')
      print('%-40s' % 'adjusted f1 before grids:', '%.5f' % adjusted_f1_before_grids)
      print('%-40s' % 'adjusted f1:', '%.5f' % adjusted_f1)

      if FLAGS.show_detail:
        print('-----------detailed f1 infos (ensemble by logits)')
        _, adjusted_f1s, class_f1s = calc_f1_alls(valid_labels, to_predict(valid_results, sum_weights))
        for i, attr in enumerate(ATTRIBUTES):
          print('%-40s' % attr, '%.5f' % adjusted_f1s[i])
        for i, cls in enumerate(CLASSES):
          print('%-40s' % cls, '%.5f' % class_f1s[i])
      print('%-40s' % 'adjusted f1 before grids:', '%.5f' % adjusted_f1_before_grids)
      print('%-40s' % 'adjusted f1:', '%.5f' % adjusted_f1)

  print('-------------------------------------OVERALL mean')
  print('ensemble by probs')
  print('%-40s' % 'f1', '%.5f' % np.mean(probs_f1_list))
  print('%-40s' % 'adjustedf f1', '%.5f' % np.mean(probs_adjusted_f1_list))
  
  print('ensemble by logits')
  print('%-40s' % 'f1:', '%.5f' % np.mean(logits_f1_list))
  print('%-40s' % 'adjusted f1:', '%.5f' % np.mean(logits_adjusted_f1_list))

  if FLAGS.grid_search:
    print('ensemble by logits after grid search')
    print('%-40s' % 'adjusted f1', '%.5f' % np.mean(grids_logits_adjusted_f1_list))

  print('-------------------------------------OVERALL recalc')
  labels = np.concatenate(labels_list, 0)
  print('ensemble by probs')
  print('%-40s' % 'f1', '%.5f' % calc_f1(labels, np.concatenate(probs_predict_list, 0)))
  print('%-40s' % 'adjustedf f1', '%.5f' % calc_f1(labels, np.concatenate(probs_adjusted_predict_list, 0)))

  print('ensemble by logits')
  print('%-40s' % 'f1:', '%.5f' % calc_f1(labels, np.concatenate(logits_predict_list, 0)))
  print('%-40s' % 'adjusted f1:', '%.5f' % calc_f1(labels, np.concatenate(logits_adjusted_predict_list, 0)))

  if FLAGS.grid_search:
    print('ensemble by logits after grid search')
    print('%-40s' % 'adjusted f1 after grid search', '%.5f' % calc_f1(labels, np.concatenate(grids_logits_adjusted_predict_list, 0)))

  _, adjusted_f1s, class_f1s = calc_f1_alls(labels, np.concatenate(logits_adjusted_predict_list, 0))
  for i, attr in enumerate(ATTRIBUTES):
    print('%-40s' % attr, '%.5f' % adjusted_f1s[i])
  for i, cls in enumerate(CLASSES):
    print('%-40s' % cls, '%.5f' % class_f1s[i])
  print('%-40s' % 'f1', '%.5f' % calc_f1(labels, np.concatenate(logits_predict_list, 0)))
  print('%-40s' % 'adjusted f1', '%.5f' % calc_f1(labels, np.concatenate(logits_adjusted_predict_list, 0)))
  if FLAGS.grid_search:
    print('%-40s' % 'adjusted f1 after grid search', '%.5f' % calc_f1(labels, np.concatenate(grids_logits_adjusted_predict_list, 0)))

  results = np.concatenate(results_list, 0)
  results = results.reshape([-1, NUM_ATTRIBUTES, NUM_CLASSES]) 
  #factor =  FLAGS.logits_factor / sum_weights
  #print('%-40s' % '* factor loss', '%.5f' % calc_loss(labels, gezi.softmax(results * factor)))
  print('%-40s' % 'loss', '%.5f' % calc_loss(labels, gezi.softmax(results)))


if __name__ == '__main__':
  tf.app.run()  
