#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read-torch-dataset.py
#        \author   chenghuige  
#          \date   2019-08-03 14:08:33.314862
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from torch.utils.data import DataLoader
import gezi
import lele

from pyt.dataset import *
from text_dataset import Dataset as TD

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def torch_(x):
  if True:
    return x
  for dim in x.shape:
    if dim == 0:
      return x

  x = x.numpy()
  # TODO..
  #if x.dtype == np.int64 or x.dtype == np.int32 or x.dtype == np.float32 or x.dtype == np.float64:
  if type(x) != np.str_:
    x = torch.from_numpy(x)
    #if torch.cuda.is_available():
      #x = x.cuda()
    x = x.to(device)

  return x

def to_torch(x, y=None):
  if True:
    for key in x:
      print(x[key])
      print(key, type(x[key][0]), type(x[key]))
      if type(x[key][0]) != np.str_:
        x[key] = x[key].to(device)
    return x, y.to(device)
  if y is not None:
    y = torch_(y)

  if not isinstance(x, dict):
    x = torch_(x)
  else:
    for key in x:
      x[key] = torch_(x[key])
  if y is None:
    return x
  else:
    return x, y
  

files = gezi.list_files('../input/valid/*')
td = TD()
ds = get_dataset(files, td)
dl = DataLoader(ds, 5, collate_fn=lele.DictPadCollate())
print(len(ds), len(dl), len(dl.dataset))
for i, (x, y) in enumerate(dl):
  print(i)
  #print('--------------', d)
  print(x['index'].shape)
  print(x['field'].shape)
  print(x['value'].shape)
  print(x['id'].shape)
  print(y.shape)
  #print(x)
  for key in x:
    print(key, type(x[key][0]), type(x[key]), x[key][0].dtype)
    
  #x, y = to_torch(x, y)
  if i == 5:
    break
