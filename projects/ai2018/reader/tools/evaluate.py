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
for row in df.iterrows():
  acc += int(row[1]['label'] == row[1]['predict'])

print('acc:', acc / len(df))
  
