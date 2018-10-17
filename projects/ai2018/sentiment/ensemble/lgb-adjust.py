#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ensemble.py
#        \author   chenghuige  
#          \date   2018-09-15 19:04:21.026718
#   \Description  now work well, worse then mean...
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
#from sklearn.preprocessing import minmax_scale
import gezi

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.metrics import f1_score 

ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
              'price_level', 'price_cost_effective', 'price_discount', 
              'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
              'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again']
num_attrs = len(ATTRIBUTES)
num_classes = 4

attr = 'others_willing_to_consume_again'
attr = 'others_overall_experience'
attr = 'location_distance_from_business_district'

def parse(l):
  if ',' in l:
    # this is list save (list of list)
    return np.array([float(x.strip()) for x in l[1:-1].split(',')])
  else:
    # this numpy save (list of numpy array)
    return np.array([float(x.strip()) for x in l[1:-1].split(' ') if x.strip()])

idx = 2

results = None

df2 = pd.DataFrame()

file_ = './ensemble.valid.debug.csv'
print(file_)
df = pd.read_csv(file_, sep=',')
df.sort_index(0, inplace=True)
labels = df.iloc[:,idx:idx+num_attrs].values
predicts = df.iloc[:,idx+num_attrs:idx+2*num_attrs].values
#scores = df['score']
scores = df['logits']
#scores = df['probs']
scores = [parse(score) for score in scores]
scores = np.array(scores)
ids = df.iloc[:,0].values 

df2['id'] = ids
for i, label in enumerate(df.columns.values[idx:idx+num_attrs]):
  df2[label] = labels[:,i]

scores = np.reshape(scores, [-1, num_attrs, num_classes])
for i, name in enumerate(ATTRIBUTES):
  #if name.startswith('others_willing_to_consume_again_'):
  if name.startswith(attr):
    for j in range(num_classes):
      df2[f'{name}_{j}'] = scores[:, i, j]

#print(df2)


i = ATTRIBUTES.index(attr)


idx = 1
x = df2.iloc[:,idx+num_attrs:].values

steps = range(10)

probs = gezi.softmax(x)

best = 0

counts = [0.020095238095238097, 0.08937142857142857, 0.2232, 0.6673333333333333]
counts = [0.796952380952381, 0.005580952380952381, 0.005076190476190476, 0.1923904761904762]
index = np.argsort(-np.array(counts))

def is_ok(factor):
  return np.sum(np.argsort(factor) == index) == 4

# for a in range(100):
#   for b in range(100):
#     for c in range(100):
#       for d in range(100):
#         factor = np.array([a + 1, b + 1, c + 1, d + 1], dtype=np.float)
#         if not is_ok(factor):
#           continue
#         preds = probs * factor
#         f1 = f1_score(labels[:,i] + 2, np.argmax(preds, 1), average='macro')
#         if f1 > best:
#           print(factor, f1)
#           best = f1
# for a in range(0,10000,100):
#   for b in range(0,10000,100):
#     for c in range(0,10000,100):
#       for d in range(0,10000,100):
#         factor = np.array([a + 1, b + 1, c + 1, d + 1], dtype=np.float)
#         if not is_ok(factor):
#           continue
#         preds = probs * factor
#         f1 = f1_score(labels[:,i] + 2, np.argmax(preds, 1), average='macro')
#         if f1 > best:
#           print(factor, f1)
#           best = f1

for a in range(0,1000,10):
  for b in range(0,1000,10):
    for c in range(0,1000,10):
      for d in range(0,1000,10):
        factor = np.array([a + 1, b + 1, c + 1, d + 1], dtype=np.float)
        if not is_ok(factor):
          continue
        preds = probs * factor
        f1 = f1_score(labels[:,i] + 2, np.argmax(preds, 1), average='macro')
        if f1 > best:
          print(factor, f1)
          best = f1


# preds = probs
# print(preds)

# print(labels[:,0] + 2)
# print(np.argmax(preds, 1))
# f1 = f1_score(labels[:,i] + 2, np.argmax(preds, 1), average='macro')

# print(f1)



