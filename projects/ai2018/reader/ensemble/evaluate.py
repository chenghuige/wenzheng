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

acc = 0
acc_if = 0
acc_wether = 0
num_if = 0
num_wehter = 0
for row in df.iterrows():
  score = int(row[1]['label'] == row[1]['predict'])
  acc += score

  if row[1]['type'] == 0:
    num_if += 1
    acc_if += score 
  else:
    num_wehter += 1
    acc_wether += score 

print('num_if', num_if, 'num_wether', num_wehter)
print('acc_if:', acc_if / num_if)
print('acc_wether:', acc_wether / num_wehter)
print('acc:', acc / len(df))
  
