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
