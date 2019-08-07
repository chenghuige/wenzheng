#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2019-08-07 11:23:23.688250
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import gezi
logging = gezi.logging

try:
  import torch 
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:
  pass

try:
  #import horovod.tensorflow as hvd
  #import horovod.torch as hvd
  #hvd.init()
  import mpi4py
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
except Exception:
  pass

def to_torch(x, y=None):
  if FLAGS.torch_only:
    for key in x:
      if type(x[key][0]) != np.str_:
        x[key] = x[key].to(device)
    return x, y.to(device)

def train(model, 
          loss_fn, 
          dataset=None,
          valid_dataset=None,
          valid_dataset2=None,
          test_dataset=None,
          evaluate_fn=None, 
          inference_fn=None,
          eval_fn=None,
          write_valid=True,
          valid_names=None,
          infer_names=None,
          infer_debug_names=None,
          valid_write_fn=None,
          infer_write_fn=None,
          valid_suffix='.valid',
          infer_suffix='.infer',
          write_streaming=False,
          optimizer=None,
          param_groups=None,
          init_fn=None,
          sep=','):
  pass
  
