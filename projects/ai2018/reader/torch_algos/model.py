#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2018-09-28 10:09:41.585876
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
from torch_algos.rnet import Rnet
from torch_algos.m_reader import MnemonicReader

# baseline
from torch_algos.baseline.model import Rnn
from torch_algos.baseline.model import MwAN 

