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

idx = 2
length = 20 

labels = df.iloc[:,idx:idx+length].values
predicts = df.iloc[:,idx+length:idx+2*length].values

f1_list = []
for i in range(length):
  f1 = f1_score(labels[:,i], predicts[:, i], average='macro')
  f1_list.append(f1)
f1 = np.mean(f1_list)

print('f1:', f1)

