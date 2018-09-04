#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   correlation.py
#        \author   chenghuige  
#          \date   2018-06-25 19:54:48.632054
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

# LCC=\frac{\sum_{i=1}^{N}(y_{i}-\overline{y})(\hat{y}_i-\overline{\hat{y}})}{\sqrt{\sum_{i=1}^{N}(y_{i}-\overline{y})^2} \sqrt{\sum_{i=1}^{N}(\hat{y}_{i}-\overline{\hat{y}})^2}}
def lcc(trues, predicts):
  true_mean = np.mean(trues)
  predict_mean = np.mean(predicts)
  up = sum([(x - true_mean) * (y - predict_mean) for x, y in zip(trues, predicts)])
  down1 = math.sqrt(sum([math.pow(x - true_mean, 2) for x in trues]))
  down2 = math.sqrt(sum([math.pow(y - predict_mean, 2) for y in predicts]))
  return up / (down1 * down2)

# SROCC=1-\frac{6\sum_{i=1}^{N}(v_i-p_i)^2}{N(N^2-1)}
def srocc(trues, predicts):
  assert len(trues) == len(predicts)
  true_args = np.argsort(trues)
  predict_args = np.argsort(predicts)
  n = len(trues)
  true_ranks = [0] * n
  predict_ranks = [0] * n
  for i, arg in enumerate(true_args):
    true_ranks[arg] = i
  for i, arg in enumerate(predict_args):
    predict_ranks[arg] = i

  down = n * (n * n - 1)
  up = 6 * sum([math.pow(x - y, 2) for x, y in zip(true_ranks, predict_ranks)])

  return 1 - up / down

    