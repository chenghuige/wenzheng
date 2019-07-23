#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2019-07-23 17:40:39.405865
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

def get_num_gpus():
  if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print("os.environ['CUDA_VISIBLE_DEVICES']", os.environ['CUDA_VISIBLE_DEVICES'])
    if os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
      return 0
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    print('CUDA_VISIBLE_DEVICES is %s'%(os.environ['CUDA_VISIBLE_DEVICES']))
    return num_gpus
  else:
    return None