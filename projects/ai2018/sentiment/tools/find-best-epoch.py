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

best_score = 0
best_epoch = None


def deal(line):
  global best_score, best_epoch
  x = line.split(' ', 5)
  epoch = int(float(x[3].split(':')[-1].split('/')[0]))
  score = float(x[-1].split(',')[0].split(':')[-1].rstrip('\'')) 
  if score > best_score:
    best_score = score
    best_epoch = epoch

for file_ in glob.glob(f'{model_dir}/log.txt*'):
  for line in open(file_):
    if 'epoch_valid' in line:
      deal(line.strip())

print('best_epoch:', best_epoch, 'best_score:', best_score)
