#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2018-09-15 19:12:03.819848
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


df = pd.read_csv(sys.argv[1])

ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
              'price_level', 'price_cost_effective', 'price_discount', 
              'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
              'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again']

idx = 2
length = 20 

labels = df.iloc[:,idx:idx+length].values
predicts = df.iloc[:,idx+length:idx+2*length].values

f1_list = []
for i in range(length):
  f1 = f1_score(labels[:,i], predicts[:, i], average='macro')
  f1_list.append(f1)
f1 = np.mean(f1_list)

for i, attr in enumerate(ATTRIBUTES):
  print(attr, f1_list[i])
print('f1:', f1)

