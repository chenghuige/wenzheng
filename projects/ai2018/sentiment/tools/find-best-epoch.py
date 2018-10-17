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

model_dir = sys.argv[1]

key = 'adjusted_f1'

if len(sys.argv) > 2:
  key = sys.argv[2]

print('key', key)

def parse(x, key='adjusted_f1'):
  idx = x.index('epoch:')
  idx2 = x.index(' ', idx)
  epoch = int(float(line[idx:idx2].split('/')[0].split(':')[1]))
  
  idx = x.index(f'{key}/mean:')
  idx2 = x.index("'", idx)
  score = float(x[idx:idx2].split(':')[-1])

  return epoch, score

best_score = 0 if key != 'loss' else 1e10
best_epoch = None

if key != 'loss':
  cmp = lambda x, y: x > y 
else:
  cmp = lambda x, y: x < y

def deal(line):
  global best_score, best_epoch
  epoch, score = parse(line, key)
  if cmp(score, best_score):
    best_score = score
    best_epoch = epoch

for file_ in glob.glob(f'{model_dir}/log.txt*'):
  for line in open(file_):
    if 'epoch_valid' in line:
      deal(line.strip())

print('best_epoch:', best_epoch, 'best_score:', best_score)
