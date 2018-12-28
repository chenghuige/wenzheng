#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-content.py
#        \author   chenghuige  
#          \date   2018-09-11 10:37:16.658519
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

files = [
         './mount/data/ai2018/sentiment/train.csv', 
         './mount/data/ai2018/sentiment/valid.csv', 
         './mount/data/ai2018/sentiment/test.csv'
        ]
            

import pandas as pd
import filter

for infile in files:
  df = pd.read_csv(infile)

  for row in df.iterrows():
    content = filter.filter(row[1][1])
    print(content)
