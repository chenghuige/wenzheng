#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   find-chars.py
#        \author   chenghuige  
#          \date   2018-10-01 20:35:40.158875
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd 
from projects.ai2018.sentiment.prepare import filter

df = pd.read_csv('/home/gezi/data/ai2018/sentiment/sentiment_classify_data/comment_raw_v2/raw_comment_v2.csv')

chars = set()
for comment in df['content']:
  comment = filter.filter(comment)
  for w in comment:
    if w not in chars:
      print(w)
      chars.add(w)

  
