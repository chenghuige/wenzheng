#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   class-info.py
#        \author   chenghuige  
#          \date   2018-09-15 16:59:15.393239
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd
import numpy as np

df = pd.read_csv(sys.argv[1])

ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
              'price_level', 'price_cost_effective', 'price_discount', 
              'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
              'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again']


counts = np.zeros([len(ATTRIBUTES), 4], dtype=np.int64)
for (_, row) in df.iterrows():
  labels = list(row[2:])
  for i, label in enumerate(labels):
    counts[i][label + 2] += 1

for attr, count in zip(ATTRIBUTES, counts):
  print(attr, [x / len(df) for x in count])