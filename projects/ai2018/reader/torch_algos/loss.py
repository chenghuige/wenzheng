#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2018-09-28 10:54:49.368829
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os

import torch

def criterion(model, x, y):
  y_ = model(x)
  loss_fn = torch.nn.CrossEntropyLoss()
  # Do not need one hot
  #y = torch.zeros(y.size(0), NUM_CLASSES, dtype=torch.int64).scatter_(1, y.view(y.size(0), 1), 1)
  loss = loss_fn(y_, y)
  
  return loss
