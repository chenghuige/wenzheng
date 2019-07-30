#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   allreduce.py
#        \author   chenghuige  
#          \date   2019-07-30 17:05:04.755084
#   \Description    nc horovodrun -np 2  python allreduce.py 
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import horovod.keras as hvd
import numpy as np
hvd.init()
hvd_r=int(hvd.rank())
assert hvd.size() == 2
#each process compute a small part of something and then compute the average etc.
test_array= np.array(range(100))
#compute a small part
span = int(100 / hvd.size())
x=np.mean(test_array[hvd_r * span: (hvd_r + 1) * span])
#x=test_array[hvd_r * span: (hvd_r + 1) * span]

#compute the average for all processes
y=hvd.allreduce(x)

#only one process print out the result
if(hvd_r==0):
  print("mean of the big array is %f"%y)  
