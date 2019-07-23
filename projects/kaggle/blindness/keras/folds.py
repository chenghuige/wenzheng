#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   folds.py
#        \author   chenghuige  
#          \date   2019-07-23 17:13:22.547651
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from sklearn.model_selection import StratifiedKFold

def get_train_valid(x, y, fold=0, num_folds=5, random_state=2019):
  skf = StratifiedKFold(n_splits=num_folds, random_state=random_state)
  
  train_index, valid_index = list(skf.split(x, y))[fold]
  return x[train_index], x[valid_index], y[train_index], y[valid_index]
