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

import pandas as pd
import glob 
import gezi 

ensembel_dir = '/home/gezi/temp/ai2018/sentiment/p40/model.csv/v11/submit.testa/'
model_dir = '/home/gezi/data3/v11/submit/'
#ensembel_dir = model_dir

valid_files = glob.glob('%s/*.valid.csv' % ensembel_dir)
infer_files = glob.glob('%s/*.infer.csv' % ensembel_dir)

valid_files = [x for x in valid_files if not 'ensemble' in x]
infer_files = [x for x in infer_files if not 'ensemble' in x]

#gpu = sys.argv[1]

#assert len(valid_files) == len(infer_files)
print('num valid fiels', len(valid_files))

num_infers = 0
for i, file_ in enumerate(valid_files):
  dest_file = file_.replace('.valid.csv', '.infer.csv')
  dest_file = os.path.join(model_dir, os.path.basename(dest_file))
  print(dest_file)
  #dest_file = file_
  #if os.path.exists(dest_file) and gezi.get_unmodify_minutes(dest_file) < 60 * 24:
  #if os.path.exists(dest_file) and gezi.get_unmodify_minutes(dest_file) < 0.0000001:
  #  print(i, dest_file, 'newly generated', gezi.get_unmodify_minutes(dest_file) / 60, 'hours')
  #if os.path.exists(dest_file) and len(pd.read_csv(dest_file)) == 2e5:
  need_infer = True
  if os.path.exists(dest_file):
    df = pd.read_csv(dest_file) 
    if len(df) == 200000:
      df = df.sort_values('id')
      df.to_csv(dest_file, index=False, encoding="utf_8_sig")
      print(i, dest_file, 'already exists with 200000 lines, you need to delete first if want to update')
      need_infer = False
  if need_infer:
    num_infers += 1 
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

    command = f'MODE=test INFER=1 SRC={src} CELL={cell} sh ./infer/v11/{script}.sh {model_dir}{file_}'
    #command = f'MODE=valid INFER=2 SRC={src} CELL={cell} c{gpu} sh ./infer/v11/{script}.sh {model_dir}{file_}'
    print(i, '-------', command)
    os.system(command) 
  
  command = f'cp {dest_file} {ensembel_dir}'
  print(i, '-----', command)
  os.system(command)
  command = f'cp {dest_file}.debug {ensembel_dir}' 
  print(i, '-----', command)
  os.system(command)

print('num_infers', num_infers)
