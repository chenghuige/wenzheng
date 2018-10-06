#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2018-09-17 20:34:23.281520
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

from algos.weights import *

import torch

from torch import nn

loss_fn = torch.nn.CrossEntropyLoss()
bloss_fn = nn.BCEWithLogitsLoss()

def criterion(model, x, y):
  y_ = model(x)
  
  #print(y.shape, y_.shape)
  # without view Expected target size (32, 4), got torch.Size([32, 20])
  loss = loss_fn(y_.view(-1, model.num_classes), y.view(-1))  

  if FLAGS.loss_type == 'add_neu_binary':
    cid = 2
    y_ = y_[:,:,cid]
    y = (y == 2).float()
    
    bloss = bloss_fn(y_, y)
    loss = loss + bloss * FLAGS.other_loss_factor

  return loss

