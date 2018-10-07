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

model_dir = '../'
thre = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7

for dir_ in glob.glob(f'{model_dir}/*/*'):
  if not os.path.isdir(dir_):
    continue
  print(dir_)
  best_score = 0
  best_epoch = None 
  for file_ in glob.glob(f'{dir_}/log.txt*'): 
    for line in open(file_):
      if 'epoch_valid' in line:
        x = line.split(' ', 5)
        epoch = int(float(x[3].split(':')[-1].split('/')[0]))
        score = float(x[-1].split(',')[0].split(':')[-1].rstrip('\''))  
        if score > best_score:
          best_score = score
          best_epoch = epoch
  print('best_epoch:', best_epoch, 'best_score:', best_score)  
  if best_epoch and best_score > thre:
    if not 'torch' in dir_:
      command = f'ensemble-cp.py {dir_}/epoch/model.ckpt-{best_epoch}'
    else:
      command = f'ensemble-cp.py {dir_}/ckpt/ckpt-{best_epoch}'
    print(command)
    os.system(command)

