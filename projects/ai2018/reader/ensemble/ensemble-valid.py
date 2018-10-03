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

import sys 
import os

import glob
import pandas as pd
import numpy as np 
import gezi

idir = sys.argv[1]


def parse(input):
  return np.array([float(x.strip()) for x in input[1:-1].split(' ') if x.strip()])


cadidates = {}
m = {}
results = {}
results2 = {}

ignore_type1s = set()
for i, file_ in enumerate(glob.glob('%s/*.valid.csv' % idir)):
  df = pd.read_csv(file_)
  #df = df.sort_index(0)
  ids = df['id'].values
  labels = df['label'].values
  candidates_ = df['candidates'].values
  predicts = df['predict'].values
  scores = df['score']
  scores = [parse(score) for score in scores]

  if not m:
    m = dict(zip(ids, labels))
    candidates = dict(zip(ids, candidates_))

  df0 = df.loc[df['type'] == 0]
  df1 = df.loc[df['type'] == 1]
  labels0 = df0['label'].values 
  predicts0 = df0['predict'].values 
  labels1 = df1['label'].values 
  predicts1 = df1['predict'].values 

  print(file_, np.mean(np.equal(labels, predicts)), np.mean(np.equal(labels0, predicts0)), np.mean(np.equal(labels1, predicts1)))

  type1_score = np.mean(np.equal(labels1, predicts1))
  
  if type1_score < 0.5:
    ignore_type1s.add(i) 
    print('ignore typ1 for', file_)
    # only use df0 
    ids = df0['id'].values 
    scores = df0['score']
    scores = [parse(score) for score in scores]
    
  for id, score in zip(ids, scores):
    if id not in results:
      results[id] = score 
    else:
      results[id] += score

  scores2 = gezi.softmax(scores, -1)
  for id, score in zip(ids, scores2):
    if id not in results2:
      results2[id] = score 
    else:
      results2[id] += score

match = 0
for id, score in results.items():
  index = np.argmax(score, -1)
  #print(id, score, index)
  predict = candidates[id].split('|')[index]
  if predict == m[id]:
    match += 1 


print('acc_by_logit', match / len(df))

match = 0
for id, score in results2.items():
  index = np.argmax(score, -1)
  predict = candidates[id].split('|')[index]
  #if predict.strip() == m[id]:
  if predict == m[id]:
    match += 1 

print('acc_by_prob', match / len(df))

cadidates = {}
results = {}
results2 = {}
for i, file_ in enumerate(glob.glob('%s/*.infer.txt.debug' % idir)):
  df = pd.read_csv(file_)
  #df = df.sort_index(0)
  ids = df['id'].values
  candidates_ = df['candidates'].values
  scores = df['score']
  scores = [parse(score) for score in scores]
  df0 = df.loc[df['type'] == 0]
  df1 = df.loc[df['type'] == 1]
  ids0 = df0['id'].values
  ids1 = df1['id'].values
  scores0 = df0['score']
  scores0 = [parse(score) for score in scores0]
  scores1 = df1['score']
  scores1 = [parse(score) for score in scores1] 

  if i in ignore_type1s:
    print('infer ignore:', file_)  
    ids = ids0
    scores = scores0 

  if not results:
    candidates = dict(zip(ids, candidates_))

  print(file_)

  scores =  gezi.softmax(scores, -1) 
  print(len(ids), len(scores))
  for id, score in zip(ids, scores):
    if id not in results:
      results[id] = score 
    else:
      results[id] += score


ofile = os.path.join(idir, 'ensemble.infer.txt')
print('ofile:', ofile)
with open(ofile, 'w') as out:
  for id, score in results.items():
    index = np.argmax(score, -1)
    #print(id, score, index)
    predict = candidates[id].split('|')[index]
    print(id, predict, sep='\t', file=out)


