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

ensembel_dir = '/home/gezi/temp/ai2018/sentiment/p40/model.csv/v11/submit/'
model_dir = '/home/gezi/data3/v11/submit/'
#ensembel_dir = model_dir

valid_files = glob.glob('%s/*.valid.csv' % ensembel_dir)
infer_files = glob.glob('%s/*.infer.csv' % ensembel_dir)

valid_files = [x for x in valid_files if not 'ensemble' in x]
infer_files = [x for x in infer_files if not 'ensemble' in x]

#assert len(valid_files) == len(infer_files)
for i, file_ in enumerate(valid_files):
  dest_file = file_.replace('.valid.csv', '.infer.csv')
  dest_file = os.path.join(model_dir, os.path.basename(dest_file))
  command = f'cp {dest_file} {ensembel_dir}'
  print(i, '-----', command)
  os.system(command)
