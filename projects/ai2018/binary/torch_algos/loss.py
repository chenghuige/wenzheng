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
import numpy as np

import lele
import torch
from torch import nn
from torch.nn import functional as F

class Criterion(object):
  def __init__(self):
    self.loss_fn = nn.BCEWithLogitsLoss()

  def forward(self, model, x, y, training=False):
    y_ = model(x) 
    y_ = y_.squeeze(1)
    y = y.float()
    loss = self.loss_fn(y_, y)

    return loss
 
