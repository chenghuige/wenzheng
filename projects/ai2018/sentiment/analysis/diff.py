#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   analyze.py
#        \author   chenghuige  
#          \date   2018-10-03 14:44:39.978269
#   \Description  

# -----------detailed f1 infos
# location_traffic_convenience 0.6712383080788638
# location_distance_from_business_district 0.5629060402359796
# location_easy_to_find 0.7248325856789342
# service_wait_time 0.6770740954491001
# service_waiters_attitude 0.8069045649815189
# service_parking_convenience 0.7526363109778337
# service_serving_speed 0.775706676245927
# price_level 0.7892694978665769
# price_cost_effective 0.7123042372602688
# price_discount 0.6898761224553072
# environment_decoration 0.7345291206372897
# environment_noise 0.7664456569694614
# environment_space 0.7672318580085105
# environment_cleaness 0.7596899305910838
# dish_portion 0.7350563643550543
# dish_taste 0.7407494293948916
# dish_look 0.6002670892812643
# dish_recommendation 0.7520857946099313
# others_overall_experience 0.5958015346685603
# others_willing_to_consume_again 0.7198627097688324
# adjusted f1_prob:[0.7167233963757594]

# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import sys 
import os

import pandas as pd 
import numpy as np 

import gezi

ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
              'price_level', 'price_cost_effective', 'price_discount', 
              'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
              'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again']

num_attrs = len(ATTRIBUTES)

classes = ['NA', 'NEG', 'NEU', 'POS']
num_classes = 4


ifile = sys.argv[1]
ifile2 = sys.argv[2]

df = pd.read_csv(ifile)
df = df.sort_values('id')

print(ifile, len(df))

df2 = pd.read_csv(ifile2)
df2 = df2.sort_values('id')

print(ifile2, len(df2))
#df = df.iloc[[0]]
#print(df)
#df2 = df2.iloc[[0]]
#print(df2)

is_valid = False 
if len(df.columns) > 2 * num_attrs:
  is_valid = True

idx = 2 if not is_valid else 2 + num_attrs
num_diff_docs = 0
num_diff_attrs = 0
for i in range(0, len(df)):
  first = True
  for j in range(num_attrs):
    cur = idx + j
    if df.iloc[i][cur] != df2.iloc[i][cur]:
      num_diff_attrs += 1
      if first:
        first = False
        print(df.iloc[i]['content'])
        num_diff_docs += 1
      attr = ATTRIBUTES[j]
      if not is_valid:
        print(attr, df.iloc[i][cur], df2.iloc[i][cur])
      else:
        print(attr, df.iloc[i][cur], df2.iloc[i][cur], df.iloc[i][cur - num_attrs])

print('num_diff_docs', num_diff_docs, file=sys.stderr)
print('num_diff_attrs', num_diff_attrs, file=sys.stderr)

