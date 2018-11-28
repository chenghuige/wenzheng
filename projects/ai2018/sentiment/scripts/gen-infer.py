#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   infer.py
#        \author   chenghuige  
#          \date   2018-11-12 23:25:26.255789
#   \Description  
# ==============================================================================
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import glob 
import gezi 

ensembel_dir = '/home/gezi/temp/ai2018/sentiment/p40/model.csv/v11/submit.testa/'
#model_dir = '/home/gezi/data3/v11/submit/'
#model_dir = '/home/gezi/data3/v11/lstm-or-sp20w/submit/' 
#model_dir = '/home/gezi/data3/v11/lstm-or-sp20w/2/submit/' 
#model_dir = '/home/gezi/data3/v11/nbert/' 
#model_dir = '/home/gezi/data3/v11/lstm-or-sp20w/3/submit/' 
#model_dir = '/home/gezi/data3/v11/nbert2/' 
#model_dir = '/home/gezi/data3/v11/bert3/' 
model_dir = '/home/gezi/data3/v11/submit.1115.2.2.2/slim/'
ensembel_dir = model_dir

valid_files = glob.glob('%s/*.valid.csv' % ensembel_dir)
infer_files = glob.glob('%s/*.infer.csv' % ensembel_dir)

valid_files = [x for x in valid_files if not 'ensemble' in x]
infer_files = [x for x in infer_files if not 'ensemble' in x]

#assert len(valid_files) == len(infer_files)
for i, file_ in enumerate(valid_files):
  file_ = os.path.basename(file_)
  file_ = gezi.strip_suffix(file_, '.valid.csv') 
  src, model = file_.split('_', 1)
  cell = 'gru'
  if 'lstm' in model:
    cell = 'lstm'

  model = model.replace('.gru', '').replace('.lstm', '')

  pattern = '_model.ckpt-'
  if pattern in model:
    script = model.split(pattern)[0]
  else:
    pattern = '_ckpt-'
    script = model.split(pattern)[0]

  #command = f'MODE=test INFER=1 SRC={src} CELL={cell} sh ./infer/v11/{script}.sh {model_dir}{file_}'
  command = f'MODE=test INFER=1 SRC={src} CELL={cell} sh ./train/v11/{script}.sh {model_dir}{file_}'
  print(command)
