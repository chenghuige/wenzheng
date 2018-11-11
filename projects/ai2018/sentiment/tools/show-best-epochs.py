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

model_dir = '..'
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
for dir_ in glob.glob(f'{model_dir}/*/*'):
  if not os.path.isdir(dir_):
    continue
  best_score = 0 if not 'loss' in key else 1e10
  best_epoch = None

  files = glob.glob(f'{dir_}/epoch/*.valid.metrics')
  if not files:
    files = glob.glob(f'{dir_}/ckpt/*.valid.metrics')

  find = False
  for file_ in files: 
    epoch = int(float(gezi.strip_suffix(file_, 'valid.metrics').split('-')[1]))
    for line in open(file_):
      name, score = line.strip().split()
      score = float(score)
      if name != key:
        continue 
      if cmp(score, best_score):
        find = True
        best_score = score
        best_epoch = epoch
  if find:
    #print(dir_)
    #print('best_epoch:', best_epoch, 'best_score:', best_score) 
    res.append((dir_.replace('../', ''), best_epoch, best_score))

  res.sort(key=lambda x: x[-1], reverse=not 'loss' in key)

for dir_, best_epoch, best_score in res:
  print('%.5f' % best_score, best_epoch, dir_)
