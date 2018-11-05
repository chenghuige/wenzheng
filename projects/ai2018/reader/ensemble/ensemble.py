#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ensemble.py
#        \author   chenghuige  
#          \date   2018-09-15 19:04:21.026718
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('debug', False, '')
flags.DEFINE_string('method', 'blend', '')
flags.DEFINE_bool('use_type', True, '')
flags.DEFINE_string('idir', '.', '')

import sys 
import os

import glob
import pandas as pd
import numpy as np 
import gezi
from tqdm import tqdm
import copy
from scipy.stats import rankdata

DEBUG = 0
idir = '.'

def parse(input):
  return np.array([float(x.strip()) for x in input[1:-1].split(' ') if x.strip()])

def blend_weights(weights, norm_factor=0.1):
  weights_min = np.min(weights)
  weights_max = np.max(weights)
  gap = weights_max - weights_min
  for i in range(len(weights)):
    weights[i] = ((weights[i] - weights_min) / gap) + norm_factor
  # ranked = rankdata(weights)
  # sum_rank = np.sum(ranked)
  # for j in range(len(weights)):
  #   weights[j] = ranked[j] / sum_rank

def main(_):
  DEBUG = FLAGS.debug
  idir = FLAGS.idir

  print('METHOD:', FLAGS.method)

  cadidates = {}
  m = {}
  results = {}
  results1 = {}
  results2 = {}

  valid_files = glob.glob(f'{idir}/*.valid.csv')
  if not DEBUG:
    print('VALID then INFER')
    infer_files = glob.glob(f'{idir}/*.infer.txt.debug')
  else:
    print('Debug mode INFER ill write result using valid ids, just for test')
    infer_files = glob.glob(f'{idir}/*.valid.csv') 

  print('num_ensembles', len(valid_files))
  print('num_infers', len(infer_files))
    
  assert len(valid_files) == len(infer_files), f'{len(valid_files)} {len(infer_files)}'

  num_ensembles = len(valid_files)

  wether_ids = None
  gdf = None
  
  weights = []
  weights_if = []
  weights_wether = []

  scores_list = []

  def get_weight(id, weight, weight_if, weight_wether):
    if FLAGS.method == 'avg' or FLAGS.method == 'mean':
      return 1.

    if not FLAGS.use_type:
      weight_ = weight 
    else:
      if id in wether_ids:
        weight_ = weight_wether
      else:
        weight_ = weight_if
        # well we use avg mean for wether_if...
        #weight_ = 1.  
    return weight_

  for i, file_ in enumerate(valid_files):
    df = pd.read_csv(file_)
    df = df.sort_values('id')
    ids = df['id'].values
    labels = df['label'].values
    if i == 0:
      gdf = df
    candidates_ = df['candidates'].values
    predicts = df['predict'].values
    scores = df['score']
    scores = [parse(score) for score in scores]
    scores = np.array(scores)
    scores_list.append(scores)
    if not m:
      m = dict(zip(ids, labels))
      candidates = dict(zip(ids, candidates_))

    df0 = df.loc[df['type'] == 0]
    df1 = df.loc[df['type'] == 1]
    labels0 = df0['label'].values 
    predicts0 = df0['predict'].values 
    labels1 = df1['label'].values 
    predicts1 = df1['predict'].values 

    if not wether_ids:
      wether_ids = set(df1['id'].values)

    acc = np.mean(np.equal(labels, predicts))
    acc_if = np.mean(np.equal(labels0, predicts0))
    acc_wether = np.mean(np.equal(labels1, predicts1))
    weights.append(acc)
    weights_if.append(acc_if)
    weights_wether.append(acc_wether)
    print(i, file_, 'acc:', acc, 'acc_if:', acc_if, 'acc_wether:', acc_wether, 'num_if:', len(df0), 'num_wether:', len(df1))


  blend_weights(weights, 1.)
  blend_weights(weights_if, 100.) # weights_if similar as disable weights
  blend_weights(weights_wether, 0.001)  # 75106
  #blend_weights(weights_wether, 0.01) #751


  print('weights', weights)
  print('weights_if', weights_if)
  print('weights_wether', weights_wether)

  for i in tqdm(range(len(valid_files)), ascii=True):    
    scores = scores_list[i]
    weight = weights[i]
    weight_if = weights_if[i]
    weight_wether = weights_wether[i]
    for id, score in zip(ids, scores):
      weight_ = get_weight(id, weight, weight_if, weight_wether)
      score *= weight_

      if id not in wether_ids:
        if id not in results:
          results[id] = copy.copy(score) 
        else:
          results[id] += score 

      if id not in results1:
        results1[id] = copy.copy(score)
      else:
        results1[id] += score 

      score = gezi.softmax(score)

      if id not in results2:
        results2[id] = copy.copy(score)
      else:
        results2[id] += score

      if id in wether_ids:
        if id not in results:
          results[id] = copy.copy(score)
        else:
          results[id] += score


  match = 0
  match_if = 0
  match_wether = 0

  for id, score in results1.items():
    index = np.argmax(score, -1)
    #print(id, score, index)
    predict = candidates[id].split('|')[index]
    match_now = 1 if predict == m[id] else 0
    match += match_now 
    if id not in wether_ids:
      match_if += match_now
    else:
      match_wether += match_now

  print('--------------by logit')
  print('acc_if_by_logit', match_if / (len(results1) - len(wether_ids)))
  print('add_wether_by_logit', match_wether / len(wether_ids))
  print('acc_by_logit', match / len(results1))


  match = 0
  match_if = 0
  match_wether = 0
  for id, score in results2.items():
    index = np.argmax(score, -1)
    predict = candidates[id].split('|')[index]
    match_now = 1 if predict == m[id] else 0
    match += match_now
    if id not in wether_ids:
      match_if += match_now
    else:
      match_wether += match_now

  print('---------------by prob')
  print('acc_if_by_prob', match_if / (len(results2) - len(wether_ids)))
  print('add_wether_by_prob', match_wether / len(wether_ids))
  print('acc_by_prob', match / len(results2))

  match = 0
  match_if = 0
  match_wether = 0
  for id, score in results.items():
    index = np.argmax(score, -1)
    predict = candidates[id].split('|')[index]
    match_now = 1 if predict == m[id] else 0
    match += match_now
    if id not in wether_ids:
      match_if += match_now
    else:
      match_wether += match_now
  
  print('--------------if by logit, wether by prob')
  print('acc_if_final', match_if / (len(results) - len(wether_ids)))
  print('add_wether_final', match_wether / len(wether_ids))
  print('acc_final', match / len(results))


  cadidates = {}
  results = {}

  for i, file_ in enumerate(infer_files):
    df = pd.read_csv(file_)
    df = df.sort_values('id')
    ids = df['id'].values
    candidates_ = df['candidates'].values
    scores = df['score']
    scores = [parse(score) for score in scores]
    scores = np.array(scores)
    df1 = df.loc[df['type'] == 1]
    ids1 = df1['id'].values

    if not wether_ids:
      wether_ids = set(ids1)

    if not results:
      candidates = dict(zip(ids, candidates_))

    print(i, file_)
    
    weight = weights[i]
    weight_if = weights_if[i]
    weight_wether = weights_wether[i]
    for id, score in zip(ids, scores):
      weight_ = get_weight(id, weight, weight_if, weight_wether)
      score = score * weight_
      
      if id not in wether_ids:
        if id not in results:
          results[id] = copy.copy(score)
        else:
          results[id] += score
      else:
        score = gezi.softmax(score)
        if id not in results:
          results[id] = copy.copy(score) 
        else:
          results[id] += score       

  ofile = os.path.join(idir, 'ensemble.infer.txt')
  print('out:', ofile)

  with open(ofile, 'w') as out:
    for id, score in results.items():
      index = np.argmax(score, -1)
      #print(id, score, index)
      predict = candidates[id].split('|')[index]
      print(id, predict, sep='\t', file=out)

  ofile = os.path.join(idir, 'ensemble.infer.debug.txt')
  print('out debug:', ofile)
  if not DEBUG:
    with open(ofile, 'w') as out:
      for id, score in results.items():
        index = np.argmax(score, -1)
        #print(id, score, index)
        predict = candidates[id].split('|')[index]
        print(id, predict, score, sep='\t', file=out)
  else:
    predicts = np.array([candidates[id].split('|')[np.argmax(score, -1)] for id, score in results.items()])
    gdf['predict'] = predicts
    print('check acc:', np.mean(np.equal(gdf['label'].values, gdf['predict'].values)))
    gdf.to_csv(ofile, index=False)

if __name__ == '__main__':
  tf.app.run()  