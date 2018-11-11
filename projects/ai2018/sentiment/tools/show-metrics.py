#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   find-best-epoch.py
#        \author   chenghuige  
#          \date   2018-10-07 10:32:35.416608
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import glob
import gezi

model_dir = '.'
if len(sys.argv) > 1:
  model_dir = sys.argv[1] 

key = 'adjusted_f1/mean'

if len(sys.argv) > 2:
  key = sys.argv[2]

print('key', key)

if not 'loss' in key:
  cmp = lambda x, y: x > y 
else:
  cmp = lambda x, y: x < y

# model.ckpt-3.00-9846.valid.metrics
# ckpt-4.valid.metrics 
res = []
for file_ in glob.glob(f'{model_dir}/*.valid.metrics'):
  epoch = int(float(gezi.strip_suffix(file_, 'valid.metrics').split('-')[1]))
  for line in open(file_):
    name, score = line.strip().split()
    score = float(score)
    if name != key:
        continue 
    res.append((gezi.strip_suffix(file_.replace('./', ''), '.valid.metrics'), epoch, score))

res.sort(key=lambda x: x[-1], reverse=not 'loss' in key)

for file_, epoch, score in res:
  print('%.5f' % score, epoch, file_)
