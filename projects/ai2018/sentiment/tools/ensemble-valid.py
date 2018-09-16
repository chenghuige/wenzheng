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
scores = np.zeros([num_attrs])

def parse(l):
  return np.array([float(x.strip()) for x in l[1:-1].split(',')])

def calc_f1(labels, predicts):
  f1_list = []
  for i in range(num_attrs):
    f1 = f1_score(labels[:,i], predicts[:, i], average='macro')
    f1_list.append(f1)
  f1 = np.mean(f1_list)
  return f1 

idx = 2
results = None
for file_ in glob.glob('%s/*.valid.csv' % idir):
  df = pd.read_csv(file_)
  labels = df.iloc[:,idx:idx+num_attrs].values
  predicts = df.iloc[:,idx+num_attrs:idx+2*num_attrs].values
  print(file_, calc_f1(labels, predicts))
  if results is None:
    results = np.zeros([len(df), num_attrs * 4])
  scores = df['score']
  for i, score in enumerate(scores):
    score = parse(score)
    #score = gezi.softmax(np.reshape(score, [num_attrs, 4]), -1)
    #score = np.reshape(score, [-1])
    results[i] += score 

results = np.reshape(results, [-1, num_attrs, 4]) 
predicts = np.argmax(results, -1) - 2

f1 = calc_f1(labels, predicts)
print('f1:', f1)

