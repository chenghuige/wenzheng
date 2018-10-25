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

model_dir = '../' if not len(sys.argv) > 1 else sys.argv[1]

thre = 0.7 if not len(sys.argv) > 2 else float(sys.argv[2])

key = 'adjusted_f1' if not len(sys.argv) > 3 else sys.argv[3]

print('model_dir', model_dir, 'thre', thre, 'key', key)

def parse(x, key='adjusted_f1'):
  idx = x.index('epoch:')
  idx2 = x.index(' ', idx)
  epoch = int(float(line[idx:idx2].split('/')[0].split(':')[1]))
  
  idx = x.index(f'{key}/mean:')
  idx2 = x.index("'", idx)
  score = float(x[idx:idx2].split(':')[-1])

  return epoch, score
  
if key != 'loss':
  cmp = lambda x, y: x > y 
else:
  cmp = lambda x, y: x < y

for dir_ in glob.glob(f'{model_dir}/*/*'):
  if not os.path.isdir(dir_):
    continue
  print(dir_)
  best_score = 0 if key != 'loss' else 1e10
  best_epoch = None
  
  for file_ in glob.glob(f'{dir_}/log.txt*'): 
    for line in open(file_):
      if 'epoch_valid' in line:
        epoch, score = parse(line, key)
        if cmp(score, best_score):
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

