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
