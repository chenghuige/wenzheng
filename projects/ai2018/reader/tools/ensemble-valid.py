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
for file_ in glob.glob('%s/*.valid.csv' % idir):
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

  print(file_, np.mean(np.equal(labels, predicts)))

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
