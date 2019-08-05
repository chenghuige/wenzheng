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
import numpy as np

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

#---------------padding input data

#https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/12

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size, dtype=vec.dtype)], dim=dim)

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        
    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max([torch.Tensor(x[0]).shape[self.dim] for x in batch])
        #print('----------', max_len)
        # pad according to max_len
        batch = [(pad_tensor(torch.Tensor(x[0]), pad=max_len, dim=self.dim), x[1]) for x in batch]
        # stack all
        xs = torch.stack([torch.Tensor(x[0]) for x in batch], dim=0)
        ys = torch.Tensor([x[1] for x in batch])
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
      
class DictPadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        
    def pad_collate(self, batch):
      ys = [None] * len(batch)
      input = {}
      ys[0] = batch[0][1]
      max_len = {}
      
      for key, val in batch[0][0].items():
        if isinstance(val, list):
          if type(val[0]) == int:
            val = torch.from_numpy(np.array(val))
          else:
            val = torch.from_numpy(np.array(val)).float()
          max_len[key] = len(val)
        input[key] = [val] * len(batch)
       
      for i in range(1, len(batch)):
        ys[i] = batch[i][1]
        for key, val in batch[i][0].items():
          if isinstance(val, list):
            if type(val[0]) == int:
              val = torch.from_numpy(np.array(val))
            else:
              val = torch.from_numpy(np.array(val)).float()
            if len(val) > max_len[key]:
              max_len[key] = len(val)
          input[key][i] = val
          
      for key, val_list in input.items():
        if key in max_len:
          for i in range(len(val_list)):
            val_list[i] = pad_tensor(val_list[i], pad=max_len[key], dim=self.dim)
            #print(i, val_list[i].shape, max_len[key])
    
          input[key] = torch.stack(val_list, dim=0)
        else:
          #... TODO why np.arry.dtype not dp.str_ but <U3 <U4 ?
          input[key] = np.array(input[key])
          if type(input[key][0]) != np.str_:
            input[key] = torch.from_numpy(input[key])
      ys = torch.from_numpy(np.array(ys))
      return input, ys
        
    def __call__(self, batch):
        return self.pad_collate(batch)
