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
from algos.config import NUM_CLASSES

import lele
import torch

from torch import nn

loss_fn = torch.nn.CrossEntropyLoss()
loss_fn2 = torch.nn.CrossEntropyLoss(reduction='none')
bloss_fn = nn.BCEWithLogitsLoss()

def criterion(model, x, y, training=False):
  y_ = model(x)
  
  #print(y.shape, y_.shape)
  # without view Expected target size (32, 4), got torch.Size([32, 20])
  if training and FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
    loss = loss_fn2(y_.view(-1, model.num_classes), y.view(-1)).view(-1, NUM_ATTRIBUTES)
    # stop some gradients due to learning_rate weights
    loss = lele.adjust_lrs(loss)
    loss = loss.mean()
  else:
    loss = loss_fn(y_.view(-1, model.num_classes), y.view(-1))  
  
  # depreciated add neu binary not help final ensemble
  if FLAGS.loss_type == 'add_neu_binary':
    cid = 2
    y_ = y_[:,:,cid]
    y = (y == 2).float()
    
    bloss = bloss_fn(y_, y)
    loss = loss + bloss * FLAGS.other_loss_factor

  return loss
 
def lm_criterion(model, x, y, training=False):
  fw_y = torch.zeros_like(y)
  bw_y = torch.zeros_like(y)
  fw_y[:, 0:-1] = y[:, 1:]
  bw_y[:, 1:] = y[:, 0:-1]

  num_targets = torch.sum((fw_y > 0).long())
  
  fw_mask = fw_y > 0
  bw_mask = bw_y > 0

  # -1 to ignore padding index 0
  fw_y = fw_y.masked_select(fw_mask) - 1
  bw_y = bw_y.masked_select(bw_mask) - 1

  y_ = model(x)
  fw_y_, bw_y_ = y_.chunk(2, -1)

  fw_y_ = fw_y_.masked_select(fw_mask.unsqueeze(-1)).view(-1, model.num_units)
  bw_y_ = bw_y_.masked_select(bw_mask.unsqueeze(-1)).view(-1, model.num_units)

  fw_y_ = model.hidden2tag(fw_y_)
  bw_y_ = model.hidden2tag(bw_y_)

  if num_targets > 0:
    fw_loss = loss_fn(fw_y_, fw_y)
    bw_loss = loss_fn(bw_y_, bw_y)
    loss = (fw_loss + bw_loss) / 2.
    loss = loss / num_targets.float()
  else:
    loss = torch.tensor(0.0).cuda()

  return loss

