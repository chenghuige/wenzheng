#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   losses.py
#        \author   chenghuige  
#          \date   2018-11-01 17:09:04.464856
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import torch

class BiLMCriterion(object):
  def __init__(self):
    self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

  def forward(self, model, x, y, training=False):
    fw_y = torch.zeros_like(y)
    bw_y = torch.zeros_like(y)
    # Notice tf not support item assignment, even eager
    fw_y[:, 0:-1] = y[:, 1:]
    bw_y[:, 1:] = y[:, 0:-1]

    # print(fw_y)
    # print(bw_y)

    num_targets = torch.sum((fw_y > 0).long())
    
    fw_mask = fw_y > 0
    bw_mask = bw_y > 0

    # -1 to ignore padding index 0
    fw_y = fw_y.masked_select(fw_mask) - 1
    bw_y = bw_y.masked_select(bw_mask) - 1

    y_ = model.encode(x, training=training)
    fw_y_, bw_y_ = y_.chunk(2, -1)

    fw_y_ = fw_y_.masked_select(fw_mask.unsqueeze(-1)).view(-1, model.num_units)
    bw_y_ = bw_y_.masked_select(bw_mask.unsqueeze(-1)).view(-1, model.num_units)

    fw_y_ = model.encode.hidden2tag(fw_y_)
    bw_y_ = model.encode.hidden2tag(bw_y_)

    if num_targets > 0:
      fw_loss = self.loss_fn(fw_y_, fw_y)
      bw_loss = self.loss_fn(bw_y_, bw_y)
      loss = (fw_loss + bw_loss) / 2.
      #print(y.shape, num_targets, fw_loss, bw_loss, loss)
      loss = loss / num_targets.float()
    else:
      loss = torch.tensor(0.0).cuda()

    return loss