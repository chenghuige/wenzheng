#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2018-10-17 06:52:08.997327
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import torch
from torch import nn
import copy
import traceback

import gezi 
logging = gezi.logging

def adjust_lrs(x, ratio=None, name='learning_rate_weights'):
  import tensorflow as tf
  if ratio is None:
    ratios = tf.get_collection(name)[-1].numpy()
    # TODO will this hurt performance ? change to use learning rate weights without tf dependence?
    ratios = torch.from_numpy(ratios).cuda()
    x = x * ratios + x.detach() * (1 - ratios)
  else:
    x = x * ratio + x.detach() * (1 - ratio)
  return x 


def load(model, path):
  try:
    checkpoint = torch.load(path)
    state = checkpoint['state_dict']   
    
    model_ = model.module if hasattr(model, 'module') else model
    new_state = {}
    for key, val in state.items():
      if key in model_.state_dict():
        new_state[key] = val

    logging.info('Updated %d keys from checkpoint %s, eopoch:%d, step:%d' % (len(new_state), path, checkpoint['epoch'], checkpoint['step']))
    new_params = model_.state_dict()
    new_params.update(new_state)
    model_.load_state_dict(new_params)
    
    model.eval()

    updated_params = []
    for name, param in model_.named_parameters():
      if name in new_state:
        updated_params.append(param)

    return checkpoint, updated_params 
  except Exception:
    logging.info(traceback.print_exc())
    return None, []

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

try:
  import torch 
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:
  pass
import numpy as np 

def torch_(x):
  for dim in x.shape:
    if dim == 0:
      return x

  #x = x.numpy()
  if x.dtype == np.int64 or x.dtype == np.int32 or x.dtype == np.float32 or x.dtype == np.float64:
    x = torch.from_numpy(x)
    #if torch.cuda.is_available():
      #x = x.cuda()
    x = x.to(device)

  return x

def to_torch(x, y=None):
  if y is not None:
    y = torch_(y)

  for key in x:
    x[key] = torch_(x[key])
  if y is None:
    return x
  else:
    return x, y
