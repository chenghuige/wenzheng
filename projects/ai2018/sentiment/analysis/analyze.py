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

flags.DEFINE_integer('id', None, '')
flags.DEFINE_string('ifile', '/home/gezi/temp/ai2018/sentiment/model/v2/0/submit6/ensemble.valid.debug.csv', '')
flags.DEFINE_string('attr', 'location_distance_from_business_district', 'location_distance_from_business_district, others_overall_experience')
flags.DEFINE_float('total', 100, '')
flags.DEFINE_string('label', None, '')
flags.DEFINE_string('key', 'score', '')

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

def parse(l):
  if ',' in l:
    return np.array([float(x.strip()) for x in l[1:-1].split(',')])
  else:
    return np.array([float(x.strip()) for x in l[1:-1].split()])


ifile = FLAGS.ifile
attr = FLAGS.attr
total = FLAGS.total

df = pd.read_csv(ifile)
df = df.sort_values('id')

idx = ATTRIBUTES.index(attr)
idx = idx * num_classes

num_oks = 0
num_errs = 0
for _, row in df.iterrows():
  label = row[attr + '_y'] + 2
  label = classes[label]
  predict = row[attr] + 2
  predict = classes[predict]
  score = row[FLAGS.key]
  score = parse(score)
  score = score[idx:idx + 4]
  prob = gezi.softmax(score)
  id = row['id']
  if FLAGS.id:
    if id != FLAGS.id:
      continue
    else:
      print(id, score)
      print(id, prob)
      print(attr, 'label  :', label)
      print(attr, 'predict:', predict)
      labels = row[2:2+num_attrs]
      labels += 2
      predicts = row[2+num_attrs:2+2*num_attrs]
      predicts += 2
      print(list(zip(ATTRIBUTES, [classes[x] for x in labels])))
      print(list(zip(ATTRIBUTES, [classes[x] for x in predicts])))
      print(row['content'])
      print(row['seg'])  
      exit(0)      

  if FLAGS.label:
    if label == FLAGS.label.upper():
      print(id, score)
      print(id, prob)
      print(attr, 'label  :', label)
      print(attr, 'predict:', predict)
      labels = row[2:2+num_attrs]
      labels += 2
      predicts = row[2+num_attrs:2+2*num_attrs]
      predicts += 2
      print(list(zip(ATTRIBUTES, [classes[x] for x in labels])))
      print(list(zip(ATTRIBUTES, [classes[x] for x in predicts])))
      print(row['content'])
      print(row['seg'])      
      num_errs += 1
  else:
    if label != predict or FLAGS.id:
      print(id, score)
      print(id, prob)
      print(attr, 'label  :', label)
      print(attr, 'predict:', predict)
      print(row['content'])
      print(row['seg'])
      num_errs += 1
    else:
      num_oks += 1

  if num_errs == total:
    print('num_oks', num_oks, 'num_errs', num_errs)
    break
    
