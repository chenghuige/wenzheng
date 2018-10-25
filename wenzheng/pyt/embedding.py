#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   embedding.py
#        \author   chenghuige  
#          \date   2018-09-29 07:39:46.517047
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import numpy as np

import torch
from torch import nn

import gezi
logging = gezi.logging

def get_embedding(vocab_size, 
                  embedding_dim=None, 
                  embedding_weight=None, 
                  trainable=True, 
                  padding_idx=0,
                  vocab2_size=0, 
                  vocab2_trainable=False):
  logging.info('vocab_size:', vocab_size, 'embedding_weight', embedding_weight)
  embedding = nn.Embedding(vocab_size,
                           embedding_dim,
                           padding_idx=padding_idx)

  if embedding_weight is not None:
    if type(embedding_weight) is str:
      if os.path.exists(embedding_weight):
        embedding_weight = np.load(embedding_weight)
      else:
        embedding_weight = None
    if embedding_weight is not None:    
      embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
  
  embedding.weight.requires_grad = trainable

  return embedding
