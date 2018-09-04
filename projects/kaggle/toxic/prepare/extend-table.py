#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   extend-table.py
#        \author   chenghuige  
#          \date   2018-02-08 17:56:56.767326
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import pandas as pd

import gezi

#m = pd.read_csv('~/data/kaggle/toxic/train.csv')
m = pd.read_csv('~/data/kaggle/toxic/test.csv')
comments = m['comment_text'].values

def process(x):
  x = gezi.filter_quota(x)
  x = gezi.tokenize(x)
  return x

comments = [process(x) for x in comments]

m['comment'] = comments

#ofile = '~/data/kaggle/toxic/train_ex.csv'
ofile = '~/data/kaggle/toxic/test_ex.csv'
m.to_csv(ofile, index=False)

