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

flags.DEFINE_bool('debug', False, '')
flags.DEFINE_string('method', 'blend', '')
flags.DEFINE_string('idir', '.', '')
flags.DEFINE_float('norm_factor', 0.0001, '')


import sys 
import os

import glob
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score 
from sklearn.preprocessing import minmax_scale
import gezi
from tqdm import tqdm

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

num_ensembles = 0

def parse(l):
  return np.array([float(x.strip()) for x in l[1:-1].split(',')])

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


def to_predict(logits):
  logits = np.reshape(logits, [-1, num_attrs, num_classes])
  probs = gezi.softmax(logits, -1)
  probs = np.reshape(probs, [-1, num_classes])
  result = np.zeros([len(probs)], dtype=int)
  for i, prob in enumerate(probs):
    # TODO try to calibrate to 0.5 ?
    if prob[0] >= 0.6:
      result[i] = -2
    else:
      result[i] = np.argmax(prob[1:]) - 1
  
  result = np.reshape(result, [-1, num_attrs])
  return result

def blend_weights(weights, norm_facotr):
  for i in range(num_attrs):
    #weights[:, i] = minmax_scale(weights[:, i])
    ws = weights[:, i]
    min_ws = np.min(ws)
    max_ws = np.max(ws)
    gap = max_ws - min_ws
    if gap > 0:
      for j in range(len(weights)):
        weights[j][i] = ((weights[j][i] - min_ws) / gap) + norm_facotr

def main(_):
  print('METHOD:', FLAGS.method)
  print('Norm factor:', FLAGS.norm_factor)
  DEBUG = FLAGS.debug 
  idir = FLAGS.idir

  # first id, sencod content ..
  idx = 2

  results = None
  results2 = None

  valid_files = glob.glob(f'{idir}/*.valid.csv')
  valid_files = [x for x in valid_files if not 'ensemble' in x]
  
  if not DEBUG:
    print('VALID then INFER')
    infer_files = glob.glob(f'{idir}/*.infer.csv.debug')
  else:
    print('Debug mode INFER ill write result using valid ids, just for test')
    infer_files = glob.glob(f'{idir}/*.valid.csv') 
    infer_files = [x for x in infer_files if not 'ensemble' in x]

  print('num_ensembles', len(valid_files))
  print('num_infers', len(infer_files))
    
  assert len(valid_files) == len(infer_files), infer_files

  num_ensembles = len(valid_files)

  weights = [] 
  scores_list = []
  for fid, file_ in enumerate(valid_files):
    df = pd.read_csv(file_)
    df= df.sort_values('id') 
    labels = df.iloc[:,idx:idx+num_attrs].values
    predicts = df.iloc[:,idx+num_attrs:idx+2*num_attrs].values
    scores = df['score']
    scores = [parse(score) for score in scores] 
    scores = np.array(scores)
    scores_list.append(scores)
    #f1 = calc_f1(labels, predicts) 
    #f1 = calc_f1(labels, to_predict(scores)) 
    #f1s = calc_f1s(labels, predicts) 
    ## to_predict better 
    f1s = calc_f1s(labels, to_predict(scores))
    print(fid, file_, calc_f1(labels, predicts), calc_f1(labels, to_predict(scores))) 
    weight = np.reshape(f1s, [num_attrs, 1])
    weights.append(weight) 

  weights = np.array(weights)
  scores_list = np.array(scores_list)

  blend_weights(weights, FLAGS.norm_factor)

  # if DEBUG:
  #   print(weights)

  for fid in tqdm(range(len(valid_files)), ascii=True):
    scores = scores_list[fid]
    if results is None:
      results = np.zeros([len(scores), num_attrs * num_classes])
      results2 = np.zeros([len(scores), num_attrs * num_classes])
    weight = weights[fid]
    if FLAGS.method == 'avg' or FLAGS.method == 'mean': 
      weight = 1.
    for i, score in enumerate(scores):
      #score_ = score
      score = np.reshape(np.reshape(score, [num_attrs, num_classes]) * weight, [-1])
      results[i] += score

      # notice softmax([1,2]) = [0.26894142, 0.73105858] softmax([2,4]) = [0.11920292, 0.88079708]
      score = gezi.softmax(np.reshape(score, [num_attrs, num_classes]), -1)

      # score = gezi.softmax(np.reshape(score_, [num_attrs, num_classes]), -1)
      # score *= weight

      score = np.reshape(score, [-1])
      results2[i] += score 

  adjusted_f1 = calc_f1(labels, to_predict(results))
  results = np.reshape(results, [-1, num_attrs, num_classes]) 
  predicts = np.argmax(results, -1) - 2
  f1 = calc_f1(labels, predicts)

  print('-----------using logits ensemble')
  print('f1:', f1)
  print('adjusted f1:', adjusted_f1)

  adjusted_f1_prob = calc_f1(labels, to_predict(results2))
  results2 = np.reshape(results2, [-1, num_attrs, num_classes]) 
  predicts2 = np.argmax(results2, -1) - 2
  f1_prob = calc_f1(labels, predicts2)

  print('-----------using prob ensemble')
  print('f1_prob:', f1_prob)
  print('adjusted f1_prob:', adjusted_f1_prob)

  print('-----------detailed f1 infos')
  _, adjusted_f1_probs, class_f1s = calc_f1_alls(labels, to_predict(results2))

  for i, attr in enumerate(ATTRIBUTES):
    print(attr, adjusted_f1_probs[i])
  for i, cls in enumerate(CLASSES):
    print(cls, class_f1s[i])


  print(f'adjusted f1_prob:[{adjusted_f1_prob}]')

  #-------------infer
  print('------------infer')
  ofile = os.path.join(idir, 'ensemble.infer.csv')
  file_ = gezi.strip_suffix(file_, '.debug')
  df = pd.read_csv(file_)

  idx = 2
  results = None
  results2 = None
  for fid, file_ in enumerate(infer_files):
    df = pd.read_csv(file_)
    df = df.sort_values('id')
    print(fid, file_)
    if results is None:
      results = np.zeros([len(df), num_attrs * num_classes])
      results2 = np.zeros([len(df), num_attrs * num_classes])
    scores = df['score']
    scores = [parse(score) for score in scores]
    scores = np.array(scores) 
    weight = weights[fid] 
    if FLAGS.method == 'avg' and FLAGS.method == 'mean': 
      weight = 1.
    for i, score in enumerate(scores):
      score = np.reshape(np.reshape(score, [num_attrs, num_classes]) * weight, [-1])
      results[i] += score
      score = gezi.softmax(np.reshape(score, [num_attrs, num_classes]), -1)
      score = np.reshape(score, [-1])
      results2[i] += score 

  predicts = to_predict(results2)

  if not DEBUG:
    columns = df.columns[idx:idx + num_attrs].values
  else:
    columns = df.columns[idx + num_attrs:idx + 2 * num_attrs].values

  if not DEBUG:
    ofile = os.path.join(idir, 'ensemble.infer.csv')
  else:
    ofile = os.path.join(idir, 'ensemble.valid.csv')

  if not DEBUG:
    file_ = gezi.strip_suffix(file_, '.debug')
    print('temp csv using for write', file_)
    df = pd.read_csv(file_)
  else:
    print('debug test using file', valid_files[-1])
    df = pd.read_csv(valid_files[-1])

  # for safe must sort id
  df = df.sort_values('id')

  # TODO better ? not using loop ?
  for i, column in enumerate(columns):
    df[column] = predicts[:, i]

  if DEBUG:
    print('check blend result', calc_f1(df.iloc[:, idx:idx + num_attrs].values, predicts))
  print(f'adjusted f1_prob:[{adjusted_f1_prob}]')


  print('out:', ofile)
  df.to_csv(ofile, index=False, encoding="utf_8_sig")


  results2 = results2 / num_ensembles
  results2 = [x for x in results2]
  df['score'] = results2 
  if not DEBUG:
    ofile = os.path.join(idir, 'ensemble.infer.debug.csv')
  else:
    ofile = os.path.join(idir, 'ensemble.valid.debug.csv')
  print('out debug:', ofile)
  df.to_csv(ofile, index=False, encoding="utf_8_sig")

if __name__ == '__main__':
  tf.app.run()  
