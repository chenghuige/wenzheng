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
from sklearn.metrics import f1_score
import gezi

idir = sys.argv[1]

num_attrs = 20

def parse(l):
  return np.array([float(x.strip()) for x in l[1:-1].split(',')])

idx = 2
results = None
for file_ in glob.glob('%s/*.infer.csv.debug' % idir):
  df = pd.read_csv(file_)
  predicts = df.iloc[:,idx:idx+num_attrs].values
  print(file_)
  if results is None:
    results = np.zeros([len(df), num_attrs * 4])
  scores = df['score']
  for i, score in enumerate(scores):
    score = parse(score)
    #score = gezi.softmax(np.reshape(score, [num_attrs, 4]), -1)
    #score = np.reshape(score, [-1])
    results[i] += score 

results = np.reshape(results, [-1, num_attrs, 4]) 

results = np.reshape(results, [-1, num_attrs, 4]) 
predicts = np.argmax(results, -1) - 2

columns = df.columns[idx:idx+num_attrs].values

ofile = os.path.join(idir, 'ensemble.infer.csv')
file_ = gezi.strip_suffix(file_, '.debug')
df = pd.read_csv(file_)

# TODO better ? not using loop ?
for i, column in enumerate(columns):
  df[column] = predicts[:, i]

print('out:', ofile)
df.to_csv(ofile, index=False, encoding="utf_8_sig")