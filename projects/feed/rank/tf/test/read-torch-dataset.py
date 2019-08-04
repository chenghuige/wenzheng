#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read-torch-dataset.py
#        \author   chenghuige  
#          \date   2019-08-03 14:08:33.314862
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from torch.utils.data import DataLoader
import gezi

from pyt.dataset import *
from text_dataset import Dataset as TD

files = gezi.list_files('../input/valid/*')
td = TD()
ds = get_dataset(files, td)
dl = DataLoader(ds, 5)
for i, d in enumerate(dl):
  print(i, d)
  if i == 3:
    break
