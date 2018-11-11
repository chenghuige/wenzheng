#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   correlations.py
#        \author   chenghuige  
#          \date   2018-10-25 11:16:14.268580
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('thre', 0.992, '')


import sys 
import os

import numpy as np
import pandas as pd 
from scipy.stats import ks_2samp

import glob

import melt
import gezi

from tqdm import tqdm

import matplotlib.pyplot as plt 
import itertools


ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
              'price_level', 'price_cost_effective', 'price_discount', 
              'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
              'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again']

os.makedirs('./bak', exist_ok=True)

#input is dir
dir = sys.argv[1] if len(sys.argv) > 1 else './'
object = None
models = []
dfs = []

df = pd.read_csv('./models.csv')
df = df[df['model'] != 'ensemble']

models_ = df['model'].values
files_ = df['file'].values 
metrics = df['adjusted_f1/mean'].values

models = []
files = []
for file, model in tqdm(zip(files_, models_), ascii=True):
  if not os.path.exists(file):
    continue
  df = pd.read_csv(file)
  df = df.sort_values('id')
  scores = [gezi.str2scores(x) for x in df['score'].values]
  scores = np.reshape(scores, [-1, len(ATTRIBUTES), 4])
  scores = gezi.softmax(scores)
  ndf = pd.DataFrame()
  ndf['score'] = np.reshape(scores, [-1])
  dfs.append(ndf)
  files.append(file)
  models.append(model)

def calc_correlation(x, y, method):
  if method.startswith('ks'):
    ks_stat, p_value = ks_2samp(x, y)
    if method == 'ks_s':
      score = ks_stat
    else:
      score = p_value  
  else:
    score = x.corr(y, method=method)
  return score

len_ = len(dfs)
c = np.zeros([len_, len_])
methods = ['pearson', 'kendall', 'spearman', 'ks_s', 'ks_p']
methods = methods[:1]
for method in methods:
  print('---------------------------------------', method)
  for i in tqdm(range(len_), ascii=True):
    for j in range(len_):
      c[i, j] = calc_correlation(dfs[i]['score'], dfs[j]['score'], method=method)    

for i in range(1, len_):
  for j in range(i + 1, len_):
    if c[i, j] > FLAGS.thre:
      print(i, j, models[i],'%.4f' % metrics[i], models[j], '%.4f' % metrics[j], c[i, j])
      if os.path.exists(files[j]) and not ('ensemble' in models[i] or 'ensemble' in models[j]):
        if glob.glob('%s_model.ckpt-*' % models[j]):
          command = 'mv %s_model.ckpt-* bak' % models[j]
          print(command)
        else:
          command = 'mv %s_ckpt-* bak' % models[j]
          print(command)
        os.system(command)
      continue

