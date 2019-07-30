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

import tensorflow as tf
import horovod.tensorflow as hvd 
from mpi4py import MPI
#import horovod.keras as hvd
import numpy as np
import melt
# Split COMM_WORLD into subcommunicators
#subcomm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.rank % 2,
#                               key=MPI.COMM_WORLD.rank)

# Initialize Horovod
#hvd.init(comm=subcomm)
hvd.init()
hvd_r=int(hvd.rank())
assert hvd.size() == 2
sess = melt.get_session()
sess.run(tf.global_variables_initializer())
#each process compute a small part of something and then compute the average etc.
test_array= np.array(range(100))
#compute a small part
span = int(100 / hvd.size())
x=test_array[hvd_r * span: (hvd_r + 1) * span]
if hvd_r == 0:
  x = list(x)
  x.append(2019)
  #x = np.array(x)
x = list(x) 
#x = [[1, a] for a in x] 
x = ['abc' for a in x]
#compute the average for all processes
#y=hvd.allgather(x, name='a')
y = MPI.COMM_WORLD.allgather(x)

#only one process print out the result
if(hvd_r==0):
  print(y)
  #print(sess.run(y))
  #print(y, len(y), sum(y))
