#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   convert.py
#        \author   chenghuige  
#          \date   2018-10-04 09:32:00.555830
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os

import pandas as pd
from tqdm import tqdm

import numpy as np

ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
              'price_level', 'price_cost_effective', 'price_discount', 
              'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
              'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again']
NUM_CLASSES = 4

df = pd.read_csv('./ratings.csv')
print(len(df))

df2 = pd.DataFrame()

ids = ['d{}'.format(i) for i in range(len(df))]
contents = df['comment'].values

df2['id'] = ids 
df2['content'] = contents
for attr in ATTRIBUTES:
  df2[attr] = [-3] * len(df)

print(len(df2))

def score2class(score):
  label = -2
  if score > 3:
    label = 1
  elif score == 3:
    label = 0
  else:
    label = -1
  return label  

labels = []
labels_env = []
labels_flavor = []
labels_service = []
for i in tqdm(range(len(df)), ascii=True):
  row = df.iloc[i]
  score = row['rating']
  score_env = row['rating_env']
  score_flavor = row['rating_flavor']
  score_service = row['rating_service']

  labels.append(score2class(score))
  labels_env.append(score2class(score_env))
  labels_flavor.append(score2class(score_flavor))
  labels_service.append(score2class(score_service))

  
df2['others_overall_experience'] = labels
df2['environment_decoration'] = labels_env
df2['dish_taste'] = labels_flavor
df2['service_waiters_attitude'] = labels_service

df2 = df2[['id', 'content'] + ATTRIBUTES]
df2.to_csv('./dianping.csv', index=False, encoding="utf_8_sig")

def main(_):
  pass

if __name__ == '__main__':
  tf.app.run()  
  
