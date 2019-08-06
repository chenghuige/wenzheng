#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2019-08-03 13:06:43.588260
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import subprocess
import linecache

import torch
from torch.utils.data import Dataset, ConcatDataset
## TODO relative path ...?
#from ..text_dataset import Dataset as TD 
#from projects.feed.rank.tf.text_dataset import Dataset as TD
#from torch.nn.utils.rnn import pack_sequence

class TextDataset(Dataset):
  def __init__(self, filename, td):
    self._filename = filename
    self._total_data = int(subprocess.check_output("wc -l " + filename, shell=True).split()[0]) 
    self.td = td 

  def __getitem__(self, idx):
    line = linecache.getline(self._filename, idx + 1)
    # lis, list, list, scalar, scalar
    feat_id, feat_field, feat_value, [label], [id] = self.td.parse_line2(line)
    ## this will use lele.DictPadCollate
    #return {'index': feat_id, 'field': feat_field, 'value': feat_value, 'id': id}, label
    ## this will use lele.DictPadCollate2 but this is slow..
    return {'index': torch.tensor(feat_id), 'field': torch.tensor(feat_field), 'value': torch.tensor(feat_value), 'id': id}, torch.tensor(label)
    
    
  def __len__(self):
    return self._total_data

def get_dataset(files, td):
  datasets = [TextDataset(x, td) for x in files]
  return ConcatDataset(datasets)


if __name__=="__main__":
  pass
