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
#from sklearn.preprocessing import minmax_scale
import gezi

idir = sys.argv[1]
ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
              'price_level', 'price_cost_effective', 'price_discount', 
              'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
              'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again']
num_attrs = len(ATTRIBUTES)

def parse(l):
  return np.array([float(x.strip()) for x in l[1:-1].split(',')])

idx = 2

results = None

df_result = pd.DataFrame()
for fid, file_ in enumerate(glob.glob('%s/*.valid.csv' % idir)):
  print(file_)
  df = pd.read_csv(file_, sep=',')
  df.sort_index(0, inplace=True)
  labels = df.iloc[:,idx:idx+num_attrs].values
  predicts = df.iloc[:,idx+num_attrs:idx+2*num_attrs].values
  scores = df['score']
  scores = [parse(score) for score in scores]
  scores = np.array(scores)
  ids = df.iloc[:,0].values 

  cname = '.'.join(os.path.basename(file_).split('.')[:-2])
  cnames = [cname + '_' + str(x) for x in range(num_attrs)]
  
  if fid == 0:
    df_result['id'] = ids
    for i, label in enumerate(df.columns.values[idx:idx+num_attrs]):
      df_result[label] = labels[:,i]
    #df.loc[df.columns.values[idx:idx+num_attrs]] = labels


  #df_result[cnames] = scores
  for i, name in enumerate(cnames):
    #print(i, name)
    df_result[name] = scores[:,i]

df_result.to_csv(f'{idir}/result.csv', index=False)


