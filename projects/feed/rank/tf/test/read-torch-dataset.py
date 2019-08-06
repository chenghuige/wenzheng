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


#import numpy as np
import horovod.torch as hvd
hvd.init()
torch.cuda.set_device(hvd.local_rank())
#seed = 1024
#torch.cuda.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

files = gezi.list_files('../input/train.small/*')

td = TD()
ds = get_dataset(files, td)
#import tensorflow as tf 

sampler = ds
sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=hvd.size(), rank=hvd.rank())

# sampler = torch.utils.data.RandomSampler(sampler)

#sampler = torch.utils.data.RandomSampler(ds)
# # seems here shuffle not work..
# sampler = torch.utils.data.distributed.DistributedSampler(
#             sampler, num_replicas=hvd.size(), rank=hvd.rank(),
#             shuffle=True)

#collate_fn = lele.DictPadCollate2()
collate_fn = lele.DictPadCollate()

dl = DataLoader(ds, 2, 
                collate_fn=collate_fn,
                sampler=sampler)

print(len(ds), len(dl), len(dl.dataset))

for epoch in range(2):
  if dl.sampler and hasattr(dl.sampler, 'set_epoch'):
    dl.sampler.set_epoch(epoch)
  for i, (x, y) in enumerate(dl):
    for j in range(len(y)):
      print('epoch', epoch, 'i', i, 'j', j, x['id'][j])

    # #print('--------------', d)
    # print(x['index'].shape, x['index'].dtype)
    # print(x['field'].shape, x['field'].dtype)
    # print(x['value'].shape, x['value'].dtype)
    # print(x['id'].shape, x['id'].dtype)
    # print(y.shape)
    # print(x)
    # for key in x:
    #   print(key, type(x[key][0]), type(x[key]), x[key][0].dtype)
      
    # if i == 2:
    #   break


# dl = DataLoader(ds, 2, 
#                 collate_fn=lele.DictPadCollate())
# import itertools
# d = itertools.cycle(dl)
# for i in range(20):
#   x, y = next(d)
#   print(x['id'])

# print('================ end')
